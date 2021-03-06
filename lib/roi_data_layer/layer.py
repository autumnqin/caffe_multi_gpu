# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue

def get_size_str( w, h):
    size_min = min(w, h)
    size_max = max(w, h)
    im_scale = float(cfg.TRAIN.SCALES[0]) / float(size_min)
    
    if round(im_scale * size_max) > cfg.TRAIN.MAX_SIZE:
        im_scale = float(cfg.TRAIN.MAX_SIZE) / float(size_max)
    im_x = int(round(w * im_scale / cfg.TRAIN.SCALE_MULTIPLE_OF) * cfg.TRAIN.SCALE_MULTIPLE_OF)
    im_y = int(round(h * im_scale / cfg.TRAIN.SCALE_MULTIPLE_OF) * cfg.TRAIN.SCALE_MULTIPLE_OF)
    return "{}x{}".format(im_x, im_y)

class RoIDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            '''
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self._perm = inds'''

            if self._groups is None:
                self._groups = {}
                for i in xrange(len(self._roidb)):
                    rd = self._roidb[i]
                    w = rd['width']
                    h = rd['height']
                    key = get_size_str(w, h)
                    if self._groups.has_key(key):
                        self._groups[key].append(i)
                    else:
                        self._groups[key] = [i]

                for (key, value) in self._groups.items():
                    if len(value) < 2048:
                        print "ratio:{} has {} images, remove all images".format(key, len(value) / 2)
                        self._groups.pop(key)

            self._group_used = {}
            for (key, value) in self._groups.items():
                self._group_used[key] = 0
                self._groups[key] = np.random.permutation(value)
                print "ratio:{} has {} images".format(key, len(value) / 2)

        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        '''
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        '''

        min_key = None
        min_ratio = 1.0
        for (key, value) in self._groups.items():
            left_ratio = float(self._group_used[key]) / float(len(value))
            if left_ratio < min_ratio:
                min_key = key
                min_ratio = left_ratio

        if min_key is None or self._group_used[min_key] + cfg.TRAIN.IMS_PER_BATCH > len(self._groups[min_key]):
            self._shuffle_roidb_inds()
            min_key = self._groups.keys()[0]

        index = self._group_used[min_key]
        db_inds = self._groups[min_key][index:index + cfg.TRAIN.IMS_PER_BATCH]
        self._group_used[min_key] = index + cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            return get_minibatch(minibatch_db, self._num_classes)

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(10)
            self._prefetch_process = BlobFetcher(self._blob_queue,
                                                 self._roidb,
                                                 self._num_classes)
            self._prefetch_process.start()
            # Terminate the child process when the parent exists
            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()
            import atexit
            atexit.register(cleanup)

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        self._groups = None

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
            max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
        self._name_to_top_map['data'] = idx
        idx += 1

        if cfg.TRAIN.HAS_RPN:
            top[idx].reshape(1, 3)
            self._name_to_top_map['im_info'] = idx
            idx += 1

            top[idx].reshape(1, 4)
            self._name_to_top_map['gt_boxes'] = idx
            idx += 1
        else: # not using RPN
            # rois blob: holds R regions of interest, each is a 5-tuple
            # (n, x1, y1, x2, y2) specifying an image batch index n and a
            # rectangle (x1, y1, x2, y2)
            top[idx].reshape(1, 5)
            self._name_to_top_map['rois'] = idx
            idx += 1

            # labels blob: R categorical labels in [0, ..., K] for K foreground
            # classes plus background
            top[idx].reshape(1)
            self._name_to_top_map['labels'] = idx
            idx += 1

            if cfg.TRAIN.BBOX_REG:
                # bbox_targets blob: R bounding-box regression targets with 4
                # targets per class
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_targets'] = idx
                idx += 1

                # bbox_inside_weights blob: At most 4 targets per roi are active;
                # thisbinary vector sepcifies the subset of active targets
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_inside_weights'] = idx
                idx += 1

                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_outside_weights'] = idx
                idx += 1

        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, queue, roidb, num_classes):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._roidb = roidb
        self._num_classes = num_classes
        self._perm = None
        self._cur = 0
        self._groups = None
        self._shuffle_roidb_inds()
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            if self._groups is None:
                self._groups = {}
                for i in xrange(len(self._roidb)):
                    rd = self._roidb[i]
                    w = rd['width']
                    h = rd['height']
                    key = get_size_str(w, h)
                    if self._groups.has_key(key):
                        self._groups[key].append(i)
                    else:
                        self._groups[key] = [i]

                for (key, value) in self._groups.items():
                    if len(value) < 2048:
                        print "ratio:{} has {} images, remove all images".format(key, len(value) / 2)
                        self._groups.pop(key)

            self._group_used = {}
            for (key, value) in self._groups.items():
                self._group_used[key] = 0
                self._groups[key] = np.random.permutation(value)
                print "ratio:{} has {} images".format(key, len(value) / 2)

        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0


    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        min_key = None
        min_ratio = 1.0
        for (key, value) in self._groups.items():
            left_ratio = float(self._group_used[key]) / float(len(value))
            if left_ratio < min_ratio:
                min_key = key
                min_ratio = left_ratio

        if min_key is None or self._group_used[min_key] + cfg.TRAIN.IMS_PER_BATCH > len(self._groups[min_key]):
            self._shuffle_roidb_inds()
            min_key = self._groups.keys()[0]

        index = self._group_used[min_key]
        db_inds = self._groups[min_key][index:index + cfg.TRAIN.IMS_PER_BATCH]
        self._group_used[min_key] = index + cfg.TRAIN.IMS_PER_BATCH
        return db_inds


    def run(self):
        print 'BlobFetcher started'
        while True:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            blobs = get_minibatch(minibatch_db, self._num_classes)
            self._queue.put(blobs)
