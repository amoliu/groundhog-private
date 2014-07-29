"""
A custom hacked trainer
"""

__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "KyungHyun Cho "
               "Caglar Gulcehre ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import numpy
import time
import logging

import theano
import theano.tensor as TT
from groundhog.utils import print_time

floatX = theano.config.floatX

logger = logging.getLogger(__name__)

class SpecialTrainer(object):
    def __init__(self,
                 model,
                 state,
                 data):
        """
        Parameters:
            :param model:
                Class describing the model used. It should provide the
                 computational graph to evaluate the model, and have a
                 similar structure to classes on the models folder
            :param state:
                Dictionary containing the current state of your job. This
                includes configuration of the job, specifically the seed,
                the startign damping factor, batch size, etc. See main.py
                for details
            :param data:
                Class describing the dataset used by the model
            :param small_emb_matrix:
                The small matrix of word representations
            :param big_emb_matrix:
                The big matrix of word representations
        """

        # Initialize member and state
        if 'adarho' not in state:
            state['adarho'] = 0.96
        if 'adaeps' not in state:
            state['adaeps'] = 1e-6

        bs = state['bs']
        self.step = 0
        self.bs = bs
        self.state = state
        self.data = data
        self.step_timer = time.time()
        self.model = model
        self.rng = numpy.random.RandomState(state['seed'])

        # Hook 0: create big matrices
        small_emb_matrix = model.small_emb_matrix
        big_emb_matrix = model.big_emb_matrix
        self.small_shape = small_emb_matrix.get_value().shape
        big_shape = big_emb_matrix.get_value(borrow=True).shape
        big_gs = numpy.zeros(big_shape, dtype=floatX)
        big_gnorm2 = numpy.zeros(big_shape, dtype=floatX)
        big_dnorm2 = numpy.zeros(big_shape, dtype=floatX)
        self.reverse_map = numpy.zeros(big_shape[0], dtype="int64")

        # Hook 1: exclude big matrix from gradient
        params = filter(lambda p : p.name != big_emb_matrix.name, model.params)

        # Constructs shared variables
        self.gs = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name)
                   for p in params]
        self.gnorm2 = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name+'_g2')
                   for p in params]
        self.dnorm2 = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name+'_d2')
                   for p in params]
        self.gdata = [theano.shared(numpy.zeros( (2,)*x.ndim,
                                                dtype=x.dtype),
                                    name=x.name) for x in model.inputs]

        # Hook 2
        # remember the auxiliary matrices corresponding to the small one
        small_gs = filter(lambda p : p.name == small_emb_matrix.name,
                self.gs)[0]
        small_gnorm2 = filter(lambda p : p.name == small_emb_matrix.name + "_g2",
                self.gnorm2)[0]
        small_dnorm2 = filter(lambda p : p.name == small_emb_matrix.name + "_d2",
                self.dnorm2)[0]
        # First slot is reversed for the parameter matrices,
        # of which the big one might be not yet loaded.
        self.small_big_pairs = [
                (None, None),
                (small_gs, big_gs),
                (small_gnorm2, big_gnorm2),
                (small_dnorm2, big_dnorm2)]

        # Compile training function
        logger.debug('Constructing grad function')
        loc_data = self.gdata
        self.prop_exprs = [x[1] for x in model.properties]
        self.prop_names = [x[0] for x in model.properties]
        self.update_rules = [x[1] for x in model.updates]
        rval = theano.clone(model.param_grads + self.update_rules + \
                            self.prop_exprs + [model.train_cost],
                            replace=zip(model.inputs, loc_data))
        nparams = len(params)
        nrules = len(self.update_rules)
        gs = rval[:nparams]
        rules = rval[nparams:nparams + nrules]
        outs = rval[nparams + nrules:]

        norm_gs = TT.sqrt(sum(TT.sum(x**2)
            for x,p in zip(gs, params) if p not in self.model.exclude_params_for_norm))
        if 'cutoff' in state and state['cutoff'] > 0:
            c = numpy.float32(state['cutoff'])
            if state['cutoff_rescale_length']:
                c = c * TT.cast(loc_data[0].shape[0], 'float32')

            notfinite = TT.or_(TT.isnan(norm_gs), TT.isinf(norm_gs))
            _gs = []
            for g,p in zip(gs, params):
                if p not in self.model.exclude_params_for_norm:
                    tmpg = TT.switch(TT.ge(norm_gs, c), g*c/norm_gs, g)
                    _gs.append(
                       TT.switch(notfinite, numpy.float32(.1)*p, tmpg))
                else:
                    _gs.append(g)
            gs = _gs
        store_gs = [(s,g) for s,g in zip(self.gs, gs)]
        updates = store_gs + [(s[0], r) for s,r in zip(model.updates, rules)]

        rho = self.state['adarho']
        eps = self.state['adaeps']

        # grad2
        gnorm2_up = [rho * gn2 + (1. - rho) * (g ** 2.) for gn2,g in zip(self.gnorm2, gs)]
        updates = updates + zip(self.gnorm2, gnorm2_up)

        logger.debug('Compiling grad function')
        st = time.time()
        self.train_fn = theano.function(
            [], outs, name='train_function',
            updates = updates,
            givens = zip(model.inputs, loc_data))
        logger.debug('took {}'.format(time.time() - st))

        self.lr = numpy.float32(1.)
        new_params = [p - (TT.sqrt(dn2 + eps) / TT.sqrt(gn2 + eps)) * g
                for p, g, gn2, dn2 in
                zip(params, self.gs, self.gnorm2, self.dnorm2)]

        updates = zip(params, new_params)
        # d2
        d2_up = [(dn2, rho * dn2 + (1. - rho) *
            (((TT.sqrt(dn2 + eps) / TT.sqrt(gn2 + eps)) * g) ** 2.))
            for dn2, gn2, g in zip(self.dnorm2, self.gnorm2, self.gs)]
        updates = updates + d2_up

        self.update_fn = theano.function(
            [], [], name='update_function',
            allow_input_downcast=True,
            updates = updates)

        self.old_cost = 1e20
        self.schedules = model.get_schedules()
        self.return_names = self.prop_names + \
                ['cost',
                        'error',
                        'time_step',
                        'whole_time', 'lr']
        self.prev_batch = None

    def __call__(self):
        # Only now we can be sure that the model is loaded:
        self.small_big_pairs[0] = (self.model.small_emb_matrix,
                self.model.big_emb_matrix.get_value(borrow=True))

        batch = self.data.next()
        assert batch

        # Hook 3:
        # renumerate the words
        g_st = time.time()
        flat_x = batch['x'].flatten()
        uniques = numpy.unique(flat_x)
        self.reverse_map[uniques] = numpy.arange(len(uniques))
        batch['x'] = self.reverse_map[flat_x].reshape(batch['x'].shape)
        # load the small matrices
        for small, big in self.small_big_pairs:
            small.set_value(big[uniques])

        # Perturb the data (! and the model)
        if isinstance(batch, dict):
            batch = self.model.perturb(**batch)
        else:
            batch = self.model.perturb(*batch)
        # Load the dataset into GPU
        # Note: not the most efficient approach in general, as it involves
        # each batch is copied individually on gpu
        if isinstance(batch, dict):
            for gdata in self.gdata:
                gdata.set_value(batch[gdata.name], borrow=True)
        else:
            for gdata, data in zip(self.gdata, batch):
                gdata.set_value(data, borrow=True)
        # Run the trianing function
        rvals = self.train_fn()
        for schedule in self.schedules:
            schedule(self, rvals[-1])
        self.update_fn()

        # Hook 4:
        # save the small matrix
        for small, big in self.small_big_pairs:
            big[uniques] = small.get_value()

        g_ed = time.time()
        self.state['lr'] = float(self.lr)
        cost = rvals[-1]
        self.old_cost = cost
        whole_time = time.time() - self.step_timer
        if self.step % self.state['trainFreq'] == 0:
            msg = '.. iter %4d cost %.3f'
            vals = [self.step, cost]
            for dx, prop in enumerate(self.prop_names):
                msg += ' '+prop+' %.2e'
                vals += [float(numpy.array(rvals[dx]))]
            msg += ' step time %s whole time %s lr %.2e'
            vals += [print_time(g_ed - g_st),
                     print_time(time.time() - self.step_timer),
                     float(self.lr)]
            logger.info(msg % tuple(vals))
        self.step += 1
        ret = dict([('cost', float(cost)),
                    ('error', float(cost)),
                       ('lr', float(self.lr)),
                       ('time_step', float(g_ed - g_st)),
                       ('whole_time', float(whole_time))]+zip(self.prop_names, rvals))
        return ret
