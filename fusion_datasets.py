import os
import shutil
import glob

def preparer_dataset_master():
    print("🚀 Démarrage de l'usine à Data V3 (Système Anti-Collision)...")
    
    # --- 1. CONFIGURATION ---
    DOSSIER_EUVP = "/content/euvp_dataset"
    DOSSIER_UIEB_RAW = "/content/uieb_data/raw" # On pointe vers le dossier parent
    DOSSIER_UIEB_REF = "/content/uieb_data/ref"
    
    DEST_RAW = "/content/dataset_complet/raw"
    DEST_REF = "/content/dataset_complet/ref"
    
    # --- 2. NETTOYAGE SÉCURISÉ ---
    if os.path.exists("/content/dataset_complet"):
        print("Suppression du dossier cible existant pour garantir une extraction propre.")
        shutil.rmtree("/content/dataset_complet")
        
    os.makedirs(DEST_RAW, exist_ok=True)
    os.makedirs(DEST_REF, exist_ok=True)

    # --- 3. INTÉGRATION DE L'UIEB (Baseline) ---
    print("\nÉtape 1 : Copie du dataset UIEB...")
    chemins_uieb_raw = glob.glob(f"{DOSSIER_UIEB_RAW}/**/*.*", recursive=True)
    chemins_uieb_ref = glob.glob(f"{DOSSIER_UIEB_REF}/**/*.*", recursive=True)
    
    # On trie pour être sûr que raw et ref correspondent
    chemins_uieb_raw.sort()
    chemins_uieb_ref.sort()
    
    compteur_uieb = 0
    for raw_path, ref_path in zip(chemins_uieb_raw, chemins_uieb_ref):
        nom_fichier = os.path.basename(raw_path)
        shutil.copy2(raw_path, os.path.join(DEST_RAW, nom_fichier))
        shutil.copy2(ref_path, os.path.join(DEST_REF, nom_fichier))
        compteur_uieb += 1
        
    print(f" {compteur_uieb} paires d'images UIEB transférées.")

    # --- 4. INTÉGRATION EUVP (Anti-Collision avec ID Unique) ---
    print("\nÉtape 2 : Extraction et renommage séquentiel du dataset EUVP...")
    
    # CRITIQUE : On force le chemin pour NE PRENDRE QUE LE DOSSIER "Paired" !
    chemins_raw_euvp = glob.glob(f"{DOSSIER_EUVP}/**/Paired/**/trainA/*.*", recursive=True)
    chemins_ref_euvp = glob.glob(f"{DOSSIER_EUVP}/**/Paired/**/trainB/*.*", recursive=True)
    
    chemins_raw_euvp.sort()
    chemins_ref_euvp.sort()
    
    compteur_euvp = 0
    for raw_path, ref_path in zip(chemins_raw_euvp, chemins_ref_euvp):
        # Création d'un ID unique sur 5 chiffres (ex: euvp_00001.jpg)
        ext = os.path.splitext(raw_path)[1] # Récupère .jpg ou .png
        nom_fichier = f"euvp_img_{compteur_euvp:05d}{ext}"
        
        shutil.copy2(raw_path, os.path.join(DEST_RAW, nom_fichier))
        shutil.copy2(ref_path, os.path.join(DEST_REF, nom_fichier))
        compteur_euvp += 1
        
    print(f" {compteur_euvp} paires d'images EUVP traitées.")
    
    # --- 5. BILAN ---
    print("\n" + "="*50)
    print("FUSION DES DATASETS TERMINÉE AVEC SUCCÈS.")
    print(f"Nombre total de paires générées : {compteur_uieb + compteur_euvp} paires.")
    print("="*50)

if __name__ == "__main__":
    preparer_dataset_master()