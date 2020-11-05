
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
from Config import Config
config_cus = Config('Config.yaml')

import tensorflow as tf
from Model import Inference_net
from data_utils import tfdata_generator_RGB
from os.path import join

imput_size = 512
batch_size = 8

class SR2HDR(object):

    def __init__(self, save_path, training=True, scale=1):
        os.system("mkdir -p " + save_path)

        self.save_path = save_path

        self.save_img_path = join(save_path, 'images')
        os.system("mkdir -p " + self.save_img_path)
        self.checkpoints_dir = join(save_path, 'model/')
        os.system("mkdir -p " + join(self.save_path, 'model'))
        self.training = training
        self.scale = scale
        self.make_model()

        config_tf = tf.ConfigProto()
        config_tf.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config_tf)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.restore(self.sess, self.saver, self.checkpoints_dir)

    def restore(self, sess, saver, checkpoints_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoints_dir)
        if ckpt:
            ckpt_files = ckpt.all_model_checkpoint_paths
            num_ckpt = len(ckpt_files)
        print(checkpoints_dir)

        if ckpt and ckpt.model_checkpoint_path and num_ckpt >= 1:
            print('的点点滴滴多多多多多多多多多多多多多多多多多多多多多多多多多多多多多多 ')
            print(ckpt_files[-1])
            saver.restore(sess, ckpt_files[-1])
            print("restore model from ", ckpt_files[-1])
            return True
        print("can not restore model from ", checkpoints_dir, 'may a new model')
        return False


    def make_model(self):

        self.hd_img = tf.placeholder(tf.float32, shape=[batch_size, imput_size, imput_size, 3], name='sdrY')
        self.sd_img = tf.placeholder(tf.float32, shape=[batch_size, imput_size, imput_size, 3], name='sdrU')

        config_gen = config_cus.HyperPara()
        self.inference_net = Inference_net(config=config_gen.Model.Generator)

        with tf.variable_scope('G') as scope:
            self.out_result = self.inference_net(self.sd_img)

        if self.training is not True:
            print('return istrain')
            return

        self.cost = tf.losses.mean_squared_error(self.out_result,self.hd_img)

        # 可变学习速率
        self.now_step = tf.placeholder(tf.int32)
        self.lr = tf.train.cosine_decay_restarts(0.0002, self.now_step, 20, t_mul=1.0, m_mul=1.0, alpha=0.0)  # 0.001
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

    def predict(self, sdr_video, save_hdr_video, width=1920, height=1080):

        feed_dict = {self.sd_img:sdr_video}

        [hdrY_pred_img] = self.sess.run([self.out_result], feed_dict=feed_dict)

        print(hdrY_pred_img)


    def train(self, tfrecord_name, num_epochs=100, num_step=50000, batch_size=batch_size, img_s=256):

        dataset = tfdata_generator_RGB(filenames=tfrecord_name, training=True, batch_size=batch_size, scale=self.scale,
                                   patchsize=(224, 224))
        iterator = dataset.make_one_shot_iterator()

        h_img, l_img = iterator.get_next()

        sess = self.sess

        for global_step in range(num_epochs):

            _lr = self.lr.eval(session=sess, feed_dict={self.now_step: global_step})

            print("------------- Epoch {:3d} --------lr: {:6f}--------------------".format(global_step, _lr))
            total_train_loss = 0


            for step in range(num_step):

                hr_raw = tf.image.resize_bilinear(h_img, (imput_size, imput_size))
                lr_raw = tf.image.resize_bilinear(l_img, (imput_size, imput_size))

                [h_img_r, l_img_r] = sess.run([hr_raw, lr_raw])


                feed_dict = {self.sd_img: h_img_r, self.hd_img: l_img_r, self.now_step: global_step}

                [train_loss,  _] = sess.run([self.cost,  self.optimizer], feed_dict=feed_dict)
                total_train_loss += train_loss
                print('current loss:', train_loss, 'current step:', step, '/', num_step)

                if step % 100 == 0:
                    print('current loss:', train_loss, 'current step:', step, '/', num_step)

            if global_step % 1 == 0:
                for i in range(5):
                    [h_img_r, l_img_r] = sess.run([h_img, l_img])

                    feed_dict = {self.sd_img: h_img_r, self.hd_img: l_img_r, self.now_step: global_step}

                    result_out_sess = sess.run(self.out_result,feed_dict=feed_dict)

                    print(result_out_sess)

            self.saver.save(sess, save_path=self.checkpoints_dir + "128x1", global_step=global_step)  # 32x3
            print("save model in ", self.checkpoints_dir)

        sess.close()


if __name__ == '__main__':

    tfpath = '/media/szx/新加卷1/HDR_NET/MODEL'

    save_path = os.path.join(tfpath, 'NODEL')

    tfrecord_name1 = '/media/szx/新加卷1/guide_denoise/waifu2x_1_60w_doc_contrast_cat_18065_10000_3744_3800_224x224_2x_video_fast_crf_35_wt_sharpen_doc_gauss_var0.00005.tfrecord'

    sdr2hdr=SR2HDR(save_path, training=True, scale=1)
    sdr2hdr.train(tfrecord_name1, num_step=240000//16, batch_size=batch_size, num_epochs=30, img_s=256)

