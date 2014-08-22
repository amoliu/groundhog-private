import cPickle

with open('/data/lisatmp3/devincol/data/translation/word_counts/combined.count.pkl') as f: 
    counts = cPickle.load(f)

sub_counts = {}
for word, count in counts.iteritems():
    subs = [s[i:j] for i in range(len(s)) for j in range(i+1, len(s)+1)]
    for s in subs:
        if s in sub_counts:
            sub_counts[s] += count
            
        else:
            sub_counts[s] = count


with open('/data/lisatmp3/devincol/data/translation/word_counts/subs.count.pkl', 'w') as f: 
    cPickle.dump(sub_counts, f)
