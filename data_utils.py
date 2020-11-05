import tensorflow as tf


def tfdata_generator_RGB(filenames, training, batch_size=16,scale=1,patchsize=(224,224)):
    '''Construct a data generator using tf.Dataset'''

    def pares_tf(example_proto):
        #定义解析的字典
        dics = {}
        dics['lr_raw'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
        dics['hr_raw'] = tf.FixedLenFeature(shape=[],dtype=tf.string)

        #调用接口解析一行样本
        parsed_example = tf.parse_single_example(serialized=example_proto,features=dics)


        hr_raw = tf.decode_raw(parsed_example['hr_raw'],out_type=tf.uint8)
        hdrY_shape=[patchsize[0]*scale, patchsize[1]*scale, 3]
        hr_raw = tf.reshape(hr_raw,shape=hdrY_shape)
        hr_raw = tf.image.convert_image_dtype(hr_raw, dtype=tf.float32)

        lr_raw = tf.decode_raw(parsed_example['lr_raw'],out_type=tf.uint8)
        hdrY_shape=[patchsize[0]*scale, patchsize[1]*scale, 3]
        lr_raw = tf.reshape(lr_raw,shape=hdrY_shape)
        lr_raw = tf.image.convert_image_dtype(lr_raw, dtype=tf.float32)

        return hr_raw,lr_raw

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    if training:
        dataset = dataset.shuffle(1000)  # depends on sample size

    # Transform and batch data at the same time
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        pares_tf, batch_size,
        num_parallel_batches=8,  # cpu cores
        drop_remainder=True if training else False))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


def tfdata_generator_SDR_HDR(filenames, training, batch_size=128,scale=1,patchsize=(160,160)):
    '''Construct a data generator using tf.Dataset'''

    def pares_tf(example_proto):
        #定义解析的字典
        dics = {}
        dics['hdrY'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
        dics['hdrU'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
        dics['hdrV'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
        dics['sdrY'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
        dics['sdrU'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
        dics['sdrV'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
        #调用接口解析一行样本
        parsed_example = tf.parse_single_example(serialized=example_proto,features=dics)
        hdrY = tf.decode_raw(parsed_example['hdrY'],out_type=tf.uint16)
        # hdrY = tf.decode_raw(parsed_example['hdrY'], out_type=tf.uint8)
        hdrY_shape=[patchsize[0]*scale, patchsize[1]*scale, 1]
        hdrY = tf.reshape(hdrY,shape=hdrY_shape)
        # hdrY = tf.image.convert_image_dtype(hdrY, dtype=tf.float32)

        hdrU = tf.decode_raw(parsed_example['hdrU'],out_type=tf.uint16)
        # hdrU = tf.decode_raw(parsed_example['hdrU'], out_type=tf.uint8)
        hdrU_shape=[patchsize[0]*scale//2, patchsize[1]*scale//2, 1]
        hdrU = tf.reshape(hdrU,shape=hdrU_shape)
        # hdrU = tf.image.convert_image_dtype(hdrU, dtype=tf.float32)

        hdrV = tf.decode_raw(parsed_example['hdrV'],out_type=tf.uint16)
        # hdrV = tf.decode_raw(parsed_example['hdrV'], out_type=tf.uint8)
        hdrV_shape=[patchsize[0]*scale//2, patchsize[1]*scale//2, 1]
        hdrV = tf.reshape(hdrV,shape=hdrV_shape)
        # hdrV = tf.image.convert_image_dtype(hdrV, dtype=tf.float32)

        sdrY = tf.decode_raw(parsed_example['sdrY'],out_type=tf.uint8)
        sdrY_shape=[patchsize[0]*scale, patchsize[1]*scale, 1]
        sdrY = tf.reshape(sdrY,shape=sdrY_shape)
        # sdrY = tf.image.convert_image_dtype(sdrY, dtype=tf.float32)

        sdrU = tf.decode_raw(parsed_example['sdrU'],out_type=tf.uint8)
        sdrU_shape=[patchsize[0]*scale//2, patchsize[1]*scale//2, 1]
        sdrU = tf.reshape(sdrU,shape=sdrU_shape)
        # sdrU = tf.image.convert_image_dtype(sdrU, dtype=tf.float32)

        sdrV = tf.decode_raw(parsed_example['sdrV'],out_type=tf.uint8)
        sdrV_shape=[patchsize[0]*scale//2, patchsize[1]*scale//2, 1]
        sdrV = tf.reshape(sdrV,shape=sdrV_shape)
        # sdrV = tf.image.convert_image_dtype(sdrV, dtype=tf.float32)

        return hdrY,hdrU,hdrV,sdrY,sdrU,sdrV

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    if training:
        dataset = dataset.shuffle(1000)  # depends on sample size

    # Transform and batch data at the same time
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        pares_tf, batch_size,
        num_parallel_batches=8,  # cpu cores
        drop_remainder=True if training else False))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset













