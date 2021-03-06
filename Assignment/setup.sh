wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh -b -p $HOME/anaconda3
./anaconda3/bin/conda init bash
. ~/.bashrc
conda create -n pytorch_env -y pytorch torchvision torchaudio numpy matplotlib tqdm -c pytorch
conda activate pytorch_env
yes | pip install sklearn
yes | pip install psrecord 
conda install -y -c conda-forge tensorboard
git clone https://github.com/drakesvoboda/eecs598/ 
export GLOO_SOCKET_IFNAME=eno1d1


psrecord "python p0.py" --plot q2.png --log q2.txt --include-children --interval .1
psrecord "python p0.py --do_chkpt" --plot q3.png --log q3.txt --include-children --interval .1
psrecord "python p0.py --batch_size 32" --plot q4-32.png --log q4-32.txt --include-children --interval .1
psrecord "python p0.py --batch_size 256" --plot q4-256.png --log q4-256.txt --include-children --interval .1

psrecord "python p1.py -n 1 -np 1 -nr 0 -a 10.10.1.1 -p 9955" --plot q6-1node.png --log q6-1node.txt --include-children --interval .1

psrecord "python p1.py -n 2 -np 1 -nr 0 -a 10.10.1.2 -p 9955" --plot q6-2node.png --log q6-2node.txt --include-children --interval .1
python p1.py -n 2 -np 1 -nr 1 -a 10.10.1.2 -p 9955

psrecord "python p1.py -n 3 -np 1 -nr 0 -a 10.10.1.1 -p 9955" --plot q6-3node.png --log q6-3node.txt --include-children --interval .1
python p1.py -n 3 -np 1 -nr 1 -a 10.10.1.1 -p 9955
python p1.py -n 3 -np 1 -nr 2 -a 10.10.1.1 -p 9955

psrecord "python p1.py -np 2" --plot q7.png --log q2.txt --include-children --interval .1
