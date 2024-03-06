
FROM quay.io/jupyter/scipy-notebook:python-3.11.6

### Environment variables
ENV GITHUB_CLI_VERSION 2.30.0

###########################
### SYSTEM INSTALLATION ###
###########################
USER root

RUN chown -R 1000:1000 /home/jovyan/work && \
    chmod -R 775 /home/jovyan/work

### System dependencies. Feel free to add packages as necessary.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Basic system usage
        lmodern=2.004.5-6.1 \
        file=1:5.41-3ubuntu0.1 \
        curl=7.81.0-1ubuntu1.15 \
        g++=4:11.2.0-1ubuntu1 \
        tmux=3.2a-4ubuntu0.2 \
        ###################################################
        ### Add your own system dependencies installed  ###
        ### with `apt-get` as needed below this comment ###
        ### Example (note the backslash after name):    ###
        ### neofetch \                                  ###
        ###################################################
        && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /tmp/library-scripts

#########################
### USER INSTALLATION ###
#########################
USER ${NB_USER}


### Anaconda (conda/mamba) packages
RUN mamba install -y -c conda-forge --freeze-installed \
        # Jupyter setup
        jupyter-server-proxy=4.1.0 \
        #######################################################
        ### Add your own conda dependencies installed with  ###
        ### `conda/mamba` as needed below this comment      ###
        ### Example (note the backslash after name):        ###
        ### scikit-learn \                                  ###
        #######################################################
        && \
    mamba clean --all

### PyPI (pip) packages
RUN pip install \ 
        nbgitpuller \
        bokeh \ 
        numpy \
        pandas \
        plotly \
        matplotlib \
        seaborn \
        transformers \ 
        tensorflow-recommenders \
        scikit-learn \
        umap-learn \
        datashader \
        holoviews \
        tensorflow \
        datasets \
        hdbscan \ 
        && \
    jupyter labextension enable nbgitpuller

### GitHub CLI Installation
RUN wget https://github.com/cli/cli/releases/download/v${GITHUB_CLI_VERSION}/gh_${GITHUB_CLI_VERSION}_linux_amd64.tar.gz -O - | \
    tar xvzf - -C /opt/conda/bin gh_${GITHUB_CLI_VERSION}_linux_amd64/bin/gh --strip-components=2

### Prints Jupyter server token when terminal is opened
RUN echo "echo \"Jupyter server token: \$(jupyter server list 2>&1 | grep -oP '(?<=token=)[[:alnum:]]*')\"" > ${HOME}/.get-jupyter-url.sh && \
    echo "sh \${HOME}/.get-jupyter-url.sh" >> ${HOME}/.bashrc
