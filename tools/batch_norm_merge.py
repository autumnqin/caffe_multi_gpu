
import sys
#os.environ['GLOG_minloglevel'] = '2'
import caffe
import argparse
import numpy as np
from caffe.proto import caffe_pb2
# from google.protobuf import text_format
from google import protobuf


class Merger(object):
    def __init__(self, output_model, output_weights):
        self.output_model_file = output_model
        self.output_weights_file = output_weights

        self.base_net = None
        self.merged_net = None

        self.base_net_param = caffe_pb2.NetParameter()
        self.merged_net_param = caffe_pb2.NetParameter()

        self.info_list = []

    def load(self, model, weights):
        with open(model, 'r') as model_file:
            protobuf.text_format.Parse(model_file.read(), self.base_net_param)
        self.base_net = caffe.Net(model, weights, caffe.TRAIN)

    def process(self):
        self._create_new_net_parameter_()
        self._create_new_weights()
        self._check_merge_()

    def _check_merge_(self):
        for layer in self.merged_net_param.layer:
            if layer.type != 'Input' and len(layer.bottom) == 0:
                print >>sys.stderr, 'The scripts works best on deploy model, by enabling after-merging check and ' \
                                    'avoiding potential problem in training.'
                return

        for input_name in self.merged_net.inputs:
            data = np.random.random(self.merged_net.blobs[input_name].data.shape)
            self.merged_net.blobs[input_name].data[...] = data.copy()
            self.base_net.blobs[input_name].data[...] = data.copy()
        base_output = self.base_net.forward()
        merged_output = self.merged_net.forward()
        diverse = 0.0
        for key in base_output:
            diverse = max(np.max(np.abs(base_output[key] - merged_output[key])), diverse)
        if diverse > 1e-5:
            print >>sys.stderr, '\nPlease note that merge convolution, batch norm & scale may corrupt the model ' \
                                'performance.\nFor this merging, the difference observed in output is %e' % diverse
        else:
            print '\nPlease note that merge convolution, batch norm & scale may corrupt the model ' \
                  'performance.\nFor this merging, the difference observed in output is %e' % diverse

    def _create_new_weights(self):
        self.merged_net = caffe.Net(self.output_model_file, caffe.TEST)
        self._copy_weights_()
        for info in self.info_list:
            self._merge_weights_(info)
        self.merged_net.save(self.output_weights_file)

    def _copy_weights_(self):
        for layer_name in self.base_net.params:
            if self.merged_net.params.has_key(layer_name):
                for j in range(len(self.base_net.params[layer_name])):
                    self.merged_net.params[layer_name][j].data[...] = self.base_net.params[layer_name][j].data.copy()

    def _merge_weights_(self, info):
        base_con = self.base_net.params[info.convolution_layer.name][0].data.astype(np.float64)
        try:
            base_con_bias = self.base_net.params[info.convolution_layer.name][1].data.astype(np.float64)
        except IndexError:
            base_con_bias = 0.0

        if info.batch_norm_layer is None:
            return
        base_bn_mean = self.base_net.params[info.batch_norm_layer.name][0].data.astype(np.float64)
        base_bn_variance = self.base_net.params[info.batch_norm_layer.name][1].data.astype(np.float64)
        base_bn_count = self.base_net.params[info.batch_norm_layer.name][2].data.astype(np.float64)[0]

        if base_bn_count == 0:
            raise Exception('Weights contains untrained batch norm layer, unable to handle it.')

        mean = base_bn_mean / base_bn_count
        std = np.power(base_bn_variance / base_bn_count + 2e-5, 0.5)

        thin_con = base_con
        for i in range(base_con.shape[0]):
            for j in range(base_con.shape[1]):
                thin_con[i, j, :, :] = base_con[i, j, :, :] / std[i]
        thin_con_bias = (- mean + base_con_bias) / std

        if info.scale_layer is not None:
            base_scale_alpha = self.base_net.params[info.scale_layer.name][0].data.astype(np.float64)
            base_scale_beta = self.base_net.params[info.scale_layer.name][1].data.astype(np.float64)
            for i in range(base_con.shape[0]):
                for j in range(base_con.shape[1]):
                    thin_con[i, j, :, :] = thin_con[i, j, :, :] * base_scale_alpha[i]
            thin_con_bias = base_scale_beta + base_scale_alpha * thin_con_bias

        self.merged_net.params[info.convolution_layer.name][0].data[...] = thin_con.copy()
        self.merged_net.params[info.convolution_layer.name][1].data[...] = thin_con_bias.copy()

    def _create_new_net_parameter_(self):
        self.merged_net_param.CopyFrom(self.base_net_param)
        self.info_list = []
        for layer in self.merged_net_param.layer:
            if layer.type == 'Convolution':
                self._handle_convlution_layer_(layer)
            elif layer.type == 'BatchNorm':
                self._handle_batch_norm_layer_(layer)
            elif layer.type == 'Scale':
                self._handle_scale_layer_(layer)
            else:
                self._handle_other_layer_(layer)
        for info in self.info_list:
            info.merge(self.merged_net_param)
        with open(self.output_model_file, 'w') as output:
            output.write(text_format.MessageToString(self.merged_net_param))

    def _handle_convlution_layer_(self, layer):
        self.info_list.append(MergeInfo(layer))

    def _handle_batch_norm_layer_(self, layer):
        to_remove_info = None
        for info in self.info_list:
            code = info.try_add_batch_norm_layer(layer)
            if code == -1:
                to_remove_info = info
                break
            elif code == 1:
                break
        if to_remove_info is not None:
            self.info_list.remove(to_remove_info)

    def _handle_scale_layer_(self, layer):
        to_remove_info = None
        for info in self.info_list:
            code = info.try_add_scale_layer(layer)
            if code == -1:
                to_remove_info = info
                break
            elif code == 1:
                break
        if to_remove_info is not None:
            self.info_list.remove(to_remove_info)

    def _handle_other_layer_(self, layer):
        to_remove_infos = []
        for info in self.info_list:
            code = info.check_with_other_layer(layer)
            if code == -1:
                to_remove_infos.append(info)
        if len(to_remove_infos) != 0:
            for info in to_remove_infos:
                self.info_list.remove(info)


class MergeInfo(object):
    def __init__(self, convolution_layer):
        self.convolution_layer = convolution_layer
        self.batch_norm_layer = None
        self.scale_layer = None
        self.top_name = convolution_layer.top[0]
        self.eps = None

    def try_add_batch_norm_layer(self, batch_norm_layer):
        if self.top_name != batch_norm_layer.bottom[0]:
            return 0

        if self.batch_norm_layer is not None or self.scale_layer is not None:
            return -1

        self.batch_norm_layer = batch_norm_layer
        self.top_name = batch_norm_layer.top[0]
        self.eps = batch_norm_layer.batch_norm_param.eps
        return 1

    def try_add_scale_layer(self, scale_layer):
        if self.top_name != scale_layer.bottom[0]:
            return 0

        if self.batch_norm_layer is None or self.scale_layer is not None:
            return -1

        self.scale_layer = scale_layer
        self.top_name = scale_layer.top[0]
        return 1

    def check_with_other_layer(self, layer):
        if self.convolution_layer.top[0] in layer.bottom and self.convolution_layer.top[0] != self.top_name:
            return -1
        elif self.batch_norm_layer is not None and self.batch_norm_layer.top[0] in layer.bottom and \
                        self.batch_norm_layer.top[0] != self.top_name:
            return -1
        else:
            return 0

    def merge(self, net_param):
        self.convolution_layer.top[0] = self.top_name
        if self.batch_norm_layer is not None:
            net_param.layer.remove(self.batch_norm_layer)
            self.convolution_layer.convolution_param.bias_term = True
            if self.scale_layer is not None:
                net_param.layer.remove(self.scale_layer)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='A script merging batch norm layer and scale layer into convolution '
                                                 'layer for Caffe model')
    parser.add_argument('-im', dest='input_model_file', required=True,
                        help='input model file, usually with .prototxt as suffix')
    parser.add_argument('-iw', dest='input_weights_file', required=True,
                        help='input weights file, usually with .caffemodel as suffix')
    parser.add_argument('-om', dest='output_model_file', required=True, help='output model file')
    parser.add_argument('-ow', dest='output_weights_file', required=True, help='output weights file')
    args = parser.parse_args()
    return args


def main(args):
    merger = Merger(args.output_model_file, args.output_weights_file)
    merger.load(args.input_model_file, args.input_weights_file)
    merger.process()

if __name__ == '__main__':
    main(parse_args())

