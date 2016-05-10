#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Caffe evaluation with selective search.

Copyright (c) 2016, Zdeněk Hřebíček

This work is based on
BVLC/Caffe web_demo example

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import time
import pickle as cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import io as StringIO
import urllib
import exifutil
import os
import sys
import cv2
from skimage import img_as_ubyte, img_as_uint, exposure, color
sys.path.append('./caffe/python')
import caffe
sys.path.append('./selectivesearch/belltailjp')
from selective_search import *
sys.path.append('./selectivesearch/AlpacaDB')
import selectivesearch as adb_selectivesearch

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) +
                               '../')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# Obtain the flask app object
app = flask.Flask(__name__, static_folder='images')


@app.route('/')
def settings():
    return flask.render_template('settings.html')


@app.route('/classification', methods=['POST'])
def classification():
    try:
        # Get all information for caffe and load it
        get_settings()
        return flask.render_template('clasification.html')
    except Exception as err:
        logging.info('Setting error: %s', err)
        return flask.render_template(
            'settings.html', has_result=True,
            result=(False, 'Make sure that all files exists.'))


# This route should be revritten to fix issue with clasification by URL with
#   button which works just after "Click for Quick examlple" FIXME
@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('image_url', '')
    try:
        string_buffer = StringIO.BytesIO(
            urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)

    except Exception as err:
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'clasification.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    # Basic clasification
    result = app.clf.classify_image(image)
    # Create list that would be embedable into html
    images_list = []
    # Run selectivesearch
    selectivesearch_result = app.clf._selective_search(image,
                                                       app.selectivesearch)
    selectivesearch_result, search_time = selectivesearch_result
    # Make output from selectivesearch embedable
    selectivesearch_result.append(image)
    embedable_images = embed_image_html(selectivesearch_result)
    x = 1
    # Fill the list with other images
    for image in embedable_images:
        images_list.append([str(x), image])
        x = x + 1
    return flask.render_template('clasification.html',
                                 images=images_list,
                                 result=result,
                                 search_time=search_time,
                                 has_result=True)


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'clasification.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )
    # Basic clasification
    result = app.clf.classify_image(image)
    # Create list that would be embedable into html
    images_list = []
    # Run selectivesearch
    selectivesearch_result = app.clf._selective_search(image,
                                                       app.selectivesearch)
    selectivesearch_result, search_time = selectivesearch_result
    # Make output from selectivesearch embedable
    selectivesearch_result.append(image)
    embedable_images = embed_image_html(selectivesearch_result)
    x = 1
    # Fill the list with other images
    for image in embedable_images:
        images_list.append([str(x), image])
        x = x + 1
    return flask.render_template('clasification.html',
                                 images=images_list,
                                 result=result,
                                 search_time=search_time,
                                 has_result=True)


def get_settings():
    # {{{ Get all settings into flask
    backend = flask.request.form['backend']
    ImagenetClassifier.default_args.update({'gpu_mode': backend})

    model_definition = flask.request.form['model_definition']
    ImagenetClassifier.default_args.update({'model_def_file':
                                            str(model_definition)})

    trained_model = flask.request.form['trained_model']
    ImagenetClassifier.default_args.update({'pretrained_model_file':
                                            str(trained_model)})

    mean_file = flask.request.form['mean_file']
    ImagenetClassifier.default_args.update({'mean_file': mean_file})

    labels = flask.request.form['labels']
    ImagenetClassifier.default_args.update({'class_labels_file': labels})

    labels = flask.request.form['hierarchy']
    ImagenetClassifier.default_args.update({'bet_file': labels})

    if flask.request.form['backend'] == "True":
        ImagenetClassifier.default_args.update({'gpu_mode': True})
    else:
        ImagenetClassifier.default_args.update({'gpu_mode': False})

    app.max_size = int(flask.request.form['max'])
    app.min_size = int(flask.request.form['min'])
    app.overlap = int(flask.request.form['overlap'])
    # This means whether to use or not to use pickle.bet file with class
    #   hierarchy
    if flask.request.form['pickle'] == "True":
        app.pickle = True
    else:
        app.pickle = False
    app.threshold = float(flask.request.form['threshold'])

    # Save selected option for selective_search for future use
    app.selectivesearch = flask.request.form['selectivesearch']
    # }}} Get all settings into flask

    # Initialize classifier + warm start by forward for allocation
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    app.clf.net.forward()


def embed_image_html(images):
    """Creates an image embedded in HTML base64 format."""
    data = []
    for image in images:
        image_pil = Image.fromarray((255 * image).astype('uint8'))
        string_buf = StringIO.BytesIO()
        image_pil.save(string_buf, format='png')
        data.append('data:image/png;base64,' +
                    string_buf.getvalue().encode('base64').replace('\n', ''))
    return data


def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS)


# This could and should be separated into it own file in the future
#   it contains image clasification and selectivesearch handling. FIXME
class ImagenetClassifier(object):
    default_args = {
        'model_def_file': (
            'data/models/bvlc_alexnet/deploy.prototxt'
            .format(REPO_DIRNAME)),
        'pretrained_model_file': (
            'data/models/bvlc_alexnet/caffenet.final.caffemodel'
            .format(REPO_DIRNAME)),
        'mean_file': (
            'caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
            .format(REPO_DIRNAME)),
        'class_labels_file': (
            'data/models/bvlc_alexnet/synset_words.txt'
            .format(REPO_DIRNAME)),
        'bet_file': (
            'data/models/bvlc_alexnet/imagenet.bet.pickle'
            .format(REPO_DIRNAME)),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255.

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, class_labels_file, bet_file, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Classifier(
            model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
            mean=np.load(mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)
        )

        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        self.labels = labels_df.sort('synset_id')['name'].values

        self.bet = cPickle.load(open(bet_file))
        # A bias to prefer children nodes in single-chain paths
        # I am setting the value to 0.1 as a quick, simple model.
        # We could use better psychological models here...
        self.bet['infogain'] -= np.array(self.bet['preferences']) * 0.1

    def classify_image(self, image):
        try:
            starttime = time.time()
            scores = self.net.predict([image], oversample=True).flatten()
            endtime = time.time()

            indices = (-scores).argsort()[:5]
            predictions = self.labels[indices]

            # In addition to the prediction text, we will also produce
            # the length for the progress bar visualization.
            meta = [
                (p, '%.5f' % scores[i])
                for i, p in zip(indices, predictions)
            ]
            logging.info('result: %s', str(meta))

            # Initialize bet results with dummy data
            bet_result = [('none', '0.000'), ('none', '0.000'),
                          ('none', '0.000'), ('none', '0.000'),
                          ('none', '0.000')]
            # Use bet pickle file only if this flag is true ...
            if app.pickle:
                # Compute expected information gain
                expected_infogain = np.dot(
                    self.bet['probmat'], scores[self.bet['idmapping']])
                expected_infogain *= self.bet['infogain']

                # sort the scores
                infogain_sort = expected_infogain.argsort()[::-1]
                bet_result = [(self.bet['words'][v], '%.5f'
                              % expected_infogain[v])
                              for v in infogain_sort[:5]]
                logging.info('bet result: %s', str(bet_result))

            return True, meta, bet_result, '%.3f' % (endtime - starttime)

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')

    def _selective_search(self, image, selectivesearch):
        starttime = time.time()
        if str(selectivesearch) == "belltailp":
            result = self.belltailp_search(image)
        elif str(selectivesearch) == "alpacadb":
            result = self.alpacadb_search(image)
        else:
            result = []
        endtime = time.time()
        # As a result we are producing time that selective search has taken
        return (result, (endtime - starttime))

    def alpacadb_search(self, image):
        image_ubyte = img_as_ubyte(image)
        candidates = set()
        # Actual selective search with some tunable settings, for more
        #   information look at it's github page
        img_lbl, regions = adb_selectivesearch.selective_search(
            image, scale=500, sigma=0.9, min_size=10)
        # Fill in candidates with same propositions as in belltailp
        for r in regions:
            # distorted rects
            x, y, w, h = r['rect']
            candidate = (x, y, w+x, h+y)
            candidates.add(candidate)
        result = self.render_valid_candidates(image, candidates)
        return result

    def belltailp_search(self, image):
        image_ubyte = img_as_ubyte(image)
        candidates = set()
        # Actual selective search with some tunable settings, for more
        #   information look at it's github page
        # feature_masks(Texture, Color, Fill, Size)
        regions = selective_search(image_ubyte,
                                   color_spaces=['rgb'],
                                   ks=[500, 100],
                                   feature_masks=[(1, 0, 0, 1), (0, 1, 1, 0)])
        # Fill in candidates
        for region in regions:
            size, (y0, x0, y1, x1) = region
            candidate = (x0, y0, x1, y1)
            candidates.add(candidate)

        result = self.render_valid_candidates(image, candidates)
        return result

    # This method filters not valid/overlayed images and renders found classes
    #   into list of images
    def render_valid_candidates(self, image, candidates):
        actual_regions = []
        (height, width, _) = image.shape
        for x, y, w, h in candidates:
            # Filter regions with bad size and propositions before actual
            #   evaluation
            if ((x-w) * (y-h)) < app.min_size:
                continue
            if ((x-w) * (y-h)) > app.max_size:
                continue
            if (x-w) / (y-h) > 1.2 or (x-w) / (y-h) > 1.2:
                continue
            # Make area of interest little bigger for better clasification
            #   results, we do this just because datasetes which are used for
            #   training models usualy contains some bacground and space around
            #   image.
            if y - 20 > 0:
                y_b = y - 20
            else:
                y_b = 0
            if x - 20 > 0:
                x_b = x - 20
            else:
                x_b = 0
            if w + 20 < width:
                w_b = w + 20
            else:
                w_b = width
            if h + 20 < height:
                h_b = h + 20
            else:
                h_b = height
            # Use CNN for clasification candidate
            classification_result = app.clf.classify_image(image[y_b:h_b,
                                                                 x_b:w_b])
            # Do not use misshapen
            if not classification_result[0]:
                continue
            # Filtration of results that clasified nothing
            if 'nan' in classification_result[1][0]:
                continue
            # Filtration of images by treshold passed from GUI
            if not (float(classification_result[1][0][1]) > app.threshold or
                    float(classification_result[2][0][1]) > app.threshold):
                continue
            # Chosing the best result
            if float(classification_result[1][0][1]) < float(classification_result[2][0][1]):
                label = classification_result[2][0][0]
                precision = float(classification_result[2][0][1])
            else:
                label = classification_result[1][0][0]
                precision = float(classification_result[1][0][1])
            # Save classification and cropp data
            actual_region = (label, precision, (x, y, w, h))
            actual_regions.append(actual_region)

        # Function for region overlap detection which returns None if
        #   rectangles don't intersect more than specified ovelap.
        def regions_overlap(regionA, regionB):
            axmin, aymin, axmax, aymax = regionA
            bxmin, bymin, bxmax, bymax = regionB
            cx = min(axmax, bxmax) - max(axmin, bxmin)
            cy = min(aymax, bymax) - max(aymin, bymin)
            if cx >= 0 and cy >= 0 and cx*cy > app.overlap:
                return True
            return False

        # Group found regions by labels to be able to iterate over classes
        from collections import defaultdict
        groups = defaultdict(list)
        for region in actual_regions:
            label, precision, crop = region
            groups[label].append((precision, crop))

        def filter_overlayed_images():
            for label in groups:
                for item in groups.get(label):
                    precision, crop = item
                    for item_2 in groups.get(label):
                        precision_2, crop_2 = item_2
                        if item == item_2:
                            continue
                        elif regions_overlap(crop, crop_2):
                            if precision >= precision_2:
                                groups[label].remove(item_2)
                            else:
                                groups[label].remove(item)
                                # Recursion
                                filter_overlayed_images()

        # To secure that there does not last more overlayed images we must call
        #   this method called recursively inside itself.
        filter_overlayed_images()

        # "Render" image classes and "bounding" boxes into list of images
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        class_images = []
        for label in groups:
            new_image = image.copy()
            # Draw all found classes bouning boxes
            for item in groups.get(label):
                precision, (x, y, w, h) = item
                new_image = cv2.rectangle(new_image, (x, y), (w, h),
                                          (255000, 255, 255), 1)
                new_image = cv2.putText(new_image, "(" + ("%.2f" % precision) +
                                        ")", (x+2, y+20),
                                        font, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
                new_image = cv2.putText(new_image, "(" + ("%.2f" % precision) +
                                        ")", (x+2, y+20),
                                        font, 1.2, (255000, 255, 255), 1,
                                        cv2.LINE_AA)
            # Write label into the image
            new_image = cv2.putText(new_image, label, (1, 22), font, 1.2,
                                    (0, 0, 0), 3, cv2.LINE_AA)
            new_image = cv2.putText(new_image, label, (1, 22), font, 1.2,
                                    (255000, 255, 255), 1, cv2.LINE_AA)
            class_images.append(new_image)
            new_image = cv2.destroyAllWindows()
        return class_images


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)

    opts, args = parser.parse_args()

    # Global settings for selective search filtration usege and other stuff...
    #   could be done differently in the future. FIXME
    # Classifier global variable definition
    app.clf = None
    # Treshold for images to be shown
    app.threshold = 1
    # Specifies type of selectivesearch
    app.selectivesearch = "noselectivesearch"
    # Max pixel size of an image
    app.max_size = 300000
    # Min pixel size of an image
    app.min_size = 500
    # Max overlap of images from same class
    app.overlap = 500
    # Usegae of xxx.bet.pickle
    app.pickle = False

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
