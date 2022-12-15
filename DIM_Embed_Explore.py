import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

for initial in ["","_initial"]:

    embedding = pk.load(open("embeddings_with_structures" + initial + ".pk","rb"))
    embedding_pure = embedding.drop("Structure",axis=1)
    print(embedding)
    embedding["transition_metal"] = embedding["Structure"].apply(lambda _:_.composition.contains_element_type("transition_metal"))
    embedding["noble_gas"] = embedding["Structure"].apply(lambda _:_.composition.contains_element_type("noble_gas"))
    embedding["metal"] = embedding["Structure"].apply(lambda _:_.composition.contains_element_type("metal"))
    embedding["halogen"] = embedding["Structure"].apply(lambda _:_.composition.contains_element_type("halogen"))
    print(embedding)

    tsne =np.transpose(TSNE().fit_transform(embedding_pure))

    plt.scatter(tsne[0],tsne[1],c=embedding["transition_metal"])
    plt.savefig("embedding_exploration/transition_metal_tsne" + initial + ".png")

    plt.scatter(tsne[0],tsne[1],c=embedding["noble_gas"])
    plt.savefig("embedding_exploration/noble_gas_tsne" + initial + ".png")

    plt.scatter(tsne[0],tsne[1],c=embedding["metal"])
    plt.savefig("embedding_exploration/metal_tsne" + initial + ".png")

    plt.scatter(tsne[0],tsne[1],c=embedding["halogen"])
    plt.savefig("embedding_exploration/halogen_tsne" + initial + ".png")
