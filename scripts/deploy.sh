#!/bin/bash
set -e

# Déploiement automatique vers le serveur Hetzner
SERVER_IP="2.28.0.161"
KEY_PATH="~/.ssh/hetzner_key"

echo "🚀 Début du déploiement vers $SERVER_IP..."

# Synchronisation des fichiers (on ignore les dossiers inutiles)
rsync -avz --exclude 'ouc_env' --exclude '__pycache__' --exclude '.git' --exclude 'results/*.jpeg' -e "ssh -i $KEY_PATH -o StrictHostKeyChecking=no" ./ root@$SERVER_IP:/root/underwater/

echo "📦 Installation et démarrage de Docker sur le serveur..."
ssh -i $KEY_PATH -o StrictHostKeyChecking=no root@$SERVER_IP << 'EOF'
  cd /root/underwater

  # Installer Docker si ce n'est pas fait
  if ! command -v docker &> /dev/null; then
      apt-get update
      apt-get install -y docker.io docker-compose
  fi

  # Lancer l'API en fond
  docker-compose up -d --build
EOF

echo "✅ Déploiement terminé ! L'API est en ligne sur http://$SERVER_IP"
