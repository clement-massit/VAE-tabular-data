import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import json
from vae import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def import_data():
    global df, X
    parser = argparse.ArgumentParser(description="Script pour traiter un fichier CSV.")
    parser.add_argument("--csv_path", type=str, required=True, help="Chemin du fichier CSV à traiter")
    
    args = parser.parse_args()
    print(f"Le fichier CSV spécifié est : {args.csv_path}")

    try:
        df = pd.read_csv(args.csv_path)
        print("Aperçu du fichier CSV :")
        print(df.head())  # Affiche les premières lignes du fichier
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV : {e}")

    return df

scaler = preprocessing.StandardScaler()
df = import_data()
X = df[['Age', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']]

def sauvegarder_hyperparametres(hyperparametres, nom_fichier="hyperparametres.json"):
    """
    Sauvegarde les hyperparamètres dans un fichier JSON.
    
    :param hyperparametres: Dictionnaire contenant les hyperparamètres.
    :param nom_fichier: Nom du fichier où sauvegarder les hyperparamètres.
    """
    with open("./output/" + nom_fichier, "w") as fichier:
        json.dump(hyperparametres, fichier, indent=4)
    print(f"Hyperparamètres sauvegardés dans {nom_fichier}.")


def main_train():
    global scaler

    latent_dim = 10  # Taille de l'espace latent
    hidden_dim = 64  # Taille des couches cachées
    batch_size = 16
    num_epochs = 20
    learning_rate = 0.001
    input_dim = X.shape[1]   # Nombre de features dans les données tabulaires
    hyperparametres = {
        "latent_dim": 10,  # Taille de l'espace latent
        "hidden_dim": 64,  # Taille des couches cachées
        "batch_size": 16,
        "num_epochs": 20,
        "learning_rate": 0.001,
        "input_dim": X.shape[1] 
    }
    sauvegarder_hyperparametres(hyperparametres, "hyperparametres.json")
    

    # Initialiser et entraîner le modèle
    encoder = VEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = VDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, input_dim=input_dim)
    vae = VAE(encoder, decoder).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    X_scaled = scaler.fit_transform(X.values)
    X_train, X_val = train_test_split(X_scaled, test_size=0.2)

    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    train_losses = []
    val_losses = []
    recon_losses = []
    kl_losses = []
    vae.train()
    for epoch in range(num_epochs):
        
        total_loss = 0
        reconstruction_loss_total = 0
        kl_loss_total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    
        for batch in progress_bar:
            x = batch[0].to(device)
            metrics = vae.train_step(batch[0], optimizer)
            total_loss += metrics['loss']
            reconstruction_loss_total += metrics['reconstruction_loss']
            kl_loss_total += metrics['kl_loss']

            progress_bar.set_postfix({"Loss": total_loss})
    

        avg_loss = total_loss / len(train_loader)
        avg_reconstruction_loss = reconstruction_loss_total / len(train_loader)
        avg_kl_loss = kl_loss_total / len(train_loader)
        train_losses.append(avg_loss)
        recon_losses.append(avg_reconstruction_loss)
        kl_losses.append(avg_kl_loss)
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader.dataset):.3f}")


        print(f'=====  epoch {epoch}/{num_epochs}, total loss {avg_loss:.4f}, recon loss {avg_reconstruction_loss:.4f}, kl_loss {avg_kl_loss:.4f}')
    torch.save(vae.state_dict(), "./output/vae_model.pth")
  
    evaluate_model(vae, val_loader, device)
    


if __name__ == "__main__":
    main_train()
    df = df
    X = X
    

    
        