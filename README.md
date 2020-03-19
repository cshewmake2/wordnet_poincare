To run the WordNet Poincare Embeddings notebook, make sure you're using Python 3.7, then please do the following:

Clone this repository into a directory of your choosing

`git clone https://github.com/cshewmake2/wordnet_poincare.git`


Change into the project directory

`cd wordnet_poincare`


Create a virtual environment for the packages to avoid conflicts

`python3 -m venv wnpe_env`


Activate the new virtual environment

`source wnpe_env/bin/activate`


Install the necessary requirements with pip

`pip install -r requirements.txt`


Create an ipython kernel for your environment

`python -m ipykernel install --user --name=wnpe_env --display-name "wnpe_env"`


Now, start a jupyer notebook server via

`jupyter notebook`


Once this is running in your browser, select the WordNet_Poincare_Embedding.ipynb file. Then, on the top of the notebook, select `Kernel/Change Kernel/wnpe_env`

Now you're ready to go with a clean start! 
