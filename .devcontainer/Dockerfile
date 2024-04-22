FROM nvcr.io/nvidia/pytorch:23.10-py3

COPY requirements.txt /requirements.txt
COPY post-create-command.sh /usr/local/bin/post-create-command.sh

RUN chmod +x /usr/local/bin/post-create-command.sh

RUN python -m pip install --upgrade pip
RUN pip install -r /requirements.txt
