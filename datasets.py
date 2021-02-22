import numpy as np
import pandas as pd
import nltk
nltk.download('wordnet')


# WORDNET
def generate_synset_dicts(synset_root):
    """Construct a collection of synset:hyponyms mappings starting from a root node."""
    synset_dicts = []
    
    def make_row(x, level = 0):
        """Create a dictionary for a given synset in order to add it to a dataframe."""
        return {'str_synset': x.name(), 'hyponyms': [hyp.name() for hyp in x.hyponyms()], 'level': level}

    def get_hyponym_tree(synset, level = 0):
        """Traverse the synset hyponym tree recursively to add new nodes."""
        if len(synset.hyponyms()) == 0:
            row = make_row(synset, level)
            row['hyponyms'] = None
            synset_dicts.append(row)
        else:
            row = make_row(synset, level)
            synset_dicts.append(row)
            [get_hyponym_tree(shi, level + 1) for shi in synset.hyponyms()]
            
    get_hyponym_tree(synset_root)
    return synset_dicts


def generate_synset_df(synset_root):
    """Construct a dataframe from the synset hyponym tree."""
    synset_dicts = generate_synset_dicts(synset_root)

    # Arrange synsets into a pandas dataframe
    df = pd.DataFrame(synset_dicts)
    df = df.drop_duplicates(subset = 'str_synset')
    df = df.sort_values(by = 'str_synset').reset_index(drop = True)
    df = df.reset_index().rename(columns = {'index': 'ind'})
    df = df.set_index('str_synset')
    df['D'] = ~df.hyponyms.isna()
    df.loc[df.D == True,'numeric_synset'] = df[df.D == True].hyponyms.apply(lambda x: [df.loc[xi].ind for xi in x])
    df.loc[df.D == False,'numeric_synset'] = None
    return df


def generate_D(df):
    """Generate a collection of relations between each u pt and its v pts."""
    df_D = df[df.D == True]
    return [pt for pt in zip(df_D.ind, df_D.numeric_synset)]

# TODO convert to torch?
def generate_Nu(D, Nu_n_samples = 10):
    """Generate a collection of random negative samples for each (u,v) in D."""
    Nu = {}
    for (u_ind, v_inds) in D:
        possible_inds = set(range(0,len(D)))
        Nu_inds = possible_inds.difference(set(v_inds))
        sample_inds = lambda x: list(np.random.choice(list(x), size = Nu_n_samples, replace = False))
        vp_inds_lists = [sample_inds(Nu_inds)+[v_ind] for i,v_ind in enumerate(v_inds)]
        #[u_ind] + [v_ind]
        Nu[u_ind] = vp_inds_lists
    return Nu