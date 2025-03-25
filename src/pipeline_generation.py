import torch
import pandas as pd
from vae import *
from pipeline_train import scaler, input_dim, hidden_dim, latent_dim, df, X, device


# df = pd.read_csv("./Sleep_Data_Sampled.csv")
# X = df.drop(columns=['Person ID', 'Gender', 'Occupation', 'Blood Pressure', 'Sleep Disorder', 'BMI Category', 'Sleep Duration'])
# df.drop(columns=['Person ID', 'Gender', 'Occupation', 'Blood Pressure', 'Sleep Disorder', 'BMI Category'])

# input_dim = X.shape[1]   # Nombre de features dans les données tabulaires
# hidden_dim = 256
# latent_dim = 64
# num_embeddings = 128
# batch_size = 32
num_samples = 10
origin_data = df
X_scaled = scaler.fit_transform(X.values)

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
    encoder = VEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = VDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, input_dim=input_dim)

    # Initialiser le modèle avec les mêmes paramètres
    model_infer = VAE(encoder,decoder)
    model_infer.load_state_dict(torch.load("./output/vae_model.pth"))
    generated_data = generate_samples(model_infer, num_samples, latent_dim)

    # Conversion en DataFrame pandas (si nécessaire)
    generated_df = pd.DataFrame(generated_data, columns=X.columns)  # Ajustez les noms de colonnes

    # Si vous avez utilisé un StandardScaler
    generated_df = pd.DataFrame(scaler.inverse_transform(generated_data), 
                            columns=X.columns)

    # og = df.drop(columns=['Person ID', 'Gender', 'Occupation', 'Blood Pressure', 'Sleep Disorder', 'BMI Category', 'Sleep Duration'])
    
    print(generated_df)
