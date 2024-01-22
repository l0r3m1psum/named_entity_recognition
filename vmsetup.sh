# Make a 20GB hd with Ubuntu 20.04 LTS

echo Insert the virtualbox additions disk, and install them.
echo Then press enter to continue.
# https://download.virtualbox.org/virtualbox/6.1.0_RC1/VBoxGuestAdditions_6.1.0_RC1.iso
read
sudo reboot
sudo usermod -aG vboxsf $USER
wget -q 'https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh'
bash Miniconda3-py38_4.11.0-Linux-x86_64.sh
rm Miniconda3-py38_4.11.0-Linux-x86_64.sh
wget -qO- 'https://get.docker.com' | sudo sh
docker image prune -a
sudo usermod -aG docker $USER
sudo reboot

# make sure to `. ~/miniconda3/bin/activate`
conda create -y -n nlp2022-hw1 python=3.9
conda activate nlp2022-hw1
conda info --envs
pip install -r requirements.txt
bash test.sh data/dev.tsv
