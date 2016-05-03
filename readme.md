Caffe evaluation with selective search
======================================

Image classification with simple object detection running as a Flask web server.

## Requirements

First thing you should do is to run this script:

    ./prepare.sh --get-all

this script will prepare proper folder structure and it will clone BVLC/Caffe repo and both AlpacaDB & belltailjp selective search algorithms. Then it will download pretrained reference mode and auxiliary files with help of script included in BVLC/Caffe.

Next thing you should do is compiling the Python Caffe interface. How-to do that is described here:
http://caffe.berkeleyvision.org/installation.html
but it can differ a little between versions of OS.

The server requires Python2 with couple of dependencies.
You can install them manually one by one or run this command:

    pip2 install -r ./web-app/dependencies.txt

If allready have caffe with compiled Python interface make sure that `/path-to-caffe/python/` is on your `PYTHONPATH`.

### Reference data

You should find reference models at `./caffe/models` and auxiliary data at `./caffe/data/ilsvrc12/` if you run preparation script.
If you did not, you can do it by yourself by running two scripts that are included in caffe:

    ./scripts/download_model_binary.py models/bvlc_reference_caffenet
    ./data/ilsvrc12/get_ilsvrc_aux.sh

Or you can use your own pretrained models.

## Running the server

Running `python2 examples/web_demo/app.py` will bring up the server, accessible at `http://127.0.0.1:5000`.
You can enable debug mode of the web server, or switch to a different port:

    % python examples/web_demo/app.py -h
    Usage: app.py [options]

    Options:
      -h, --help            show this help message and exit
      -d, --debug           enable debug mode
      -p PORT, --port=PORT  which port to serve content on

 Usage of web interface should be self descriptive. If not you can contact me and I will try to include some usage.

# License

This implementation is publicly available under the Apache License, Version 2.0. See LICENSE.txt for more details.

# References

This work is based on BVLC/Caffe web_demo example, which was just for simple evaluation of caffe models not selective search.

\[1\] <a name="jia2014caffe"> [Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor: Caffe: Convolutional Architecture for Fast Feature Embedding](https://github.com/BVLC/caffe) <br/>
\[2\] https://github.com/belltailjp/selective_search_py<br/>
\[3\] https://github.com/AlpacaDB/selectivesearch
