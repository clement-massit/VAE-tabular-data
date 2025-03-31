import torch
import pandas as pd
import json
from vae import *
from pipeline_train import scaler, device, df, X, shape_processor, pipe, preprocessor

# input_dim, hidden_dim, latent_dim, df, X


# df = pd.read_csv("./Sleep_Data_Sampled.csv")
# X = df.drop(columns=['Person ID', 'Gender', 'Occupation', 'Blood Pressure', 'Sleep Disorder', 'BMI Category', 'Sleep Duration'])
# df.drop(columns=['Person ID', 'Gender', 'Occupation', 'Blood Pressure', 'Sleep Disorder', 'BMI Category'])

# input_dim = X.shape[1]   # Nombre de features dans les données tabulaires
# hidden_dim = 256
# latent_dim = 64
# num_embeddings = 128
# batch_size = 32


def get_hyperparametres_json(nom_fichier):
    """
    Lit les hyperparamètres depuis un fichier JSON.
    
    :param nom_fichier: Nom du fichier JSON contenant les hyperparamètres.
    :return: Dictionnaire des hyperparamètres.
    """
    with open(nom_fichier, "r") as fichier:
        hyperparametres = json.load(fichier)
    print(f"Hyperparamètres chargés depuis {nom_fichier}:")
    print(hyperparametres)
    return hyperparametres

def inverse_transform_pipe(X):
    arrays = []
    for name, indices in pipe['preprocessor'].output_indices_.items():
    
        transformer = pipe['preprocessor'].named_transformers_.get(name, None)
        arr = X[:, indices.start:indices.stop]
        if transformer is not None and hasattr(transformer, 'inverse_transform'):
            arr = transformer.inverse_transform(arr)
        arrays.append(arr)
    return np.concatenate(arrays, axis=1)

# Exemple d'utilisation
hyperparametres = get_hyperparametres_json("./output/hyperparametres.json")

latent_dim = hyperparametres["latent_dim"]  # Taille de l'espace latent
hidden_dim = hyperparametres["hidden_dim"]  # Taille des couches cachées
batch_size = hyperparametres["batch_size"]
num_epochs = hyperparametres["num_epochs"]
learning_rate = hyperparametres["learning_rate"]
input_dim = hyperparametres["input_dim"]

num_samples = 10
origin_data = df

def generate_samples(model, num_samples, latent_dim, device=device):
    """
    Génère de nouveaux échantillons à partir de l'espace latent
    """
    model.eval()
    with torch.no_grad():
        # Génération de vecteurs latents aléatoires
        z = torch.randn(num_samples, latent_dim).to(device)
        
        # Passage par le décodeur
        generated_samples = model.decoder(z)
    
    return generated_samples.cpu().numpy()

if __name__ == "__main__":
    encoder = VEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, processor_dim=shape_processor)
    decoder = VDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, input_dim=input_dim, processor_dim=shape_processor)

    # Initialiser le modèle avec les mêmes paramètres
    model_infer = VAE(encoder,decoder)
    model_infer.load_state_dict(torch.load("./output/vae_model.pth"))


    generated_data = generate_samples(model_infer, num_samples, shape_processor)
    og = pd.DataFrame(pipe['preprocessor'].transform(df),
                               columns=preprocessor.get_feature_names_out().tolist())

    preprocessor_transformer = pipe.named_steps['preprocessor']

    df_inverse_transformed = inverse_transform_pipe(generated_data)
    generated_df = pd.DataFrame(df_inverse_transformed, columns=[col for cat_num in [p[2] for p in preprocessor.get_params()['transformers']] for col in cat_num])


    # Conversion en DataFrame pandas (si nécessaire)
    # generated_df = pd.DataFrame(generated_data, columns=X.columns)  # Ajustez les noms de colonnes

    # # Si vous avez utilisé un StandardScaler
    # generated_df = pd.DataFrame(scaler.inverse_transform(generated_data), 
    #                         columns=X.columns)

    # og = df.drop(columns=['Person ID', 'Gender', 'Occupation', 'Blood Pressure', 'Sleep Disorder', 'BMI Category', 'Sleep Duration'])
    
    print(generated_df)
