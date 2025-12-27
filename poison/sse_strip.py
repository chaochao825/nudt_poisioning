import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 仅显示错误
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN优化
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import utils as np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import warnings
import math
import random
import time
import scipy
from utils import sse_envelope, RunSummary, set_summary_writer
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)
# 抑制所有警告
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

def emit(event: str, data: dict, progress: float = None, log: str = None):
    message = data.get("message", event)
    details = {k: v for k, v in data.items() if k != "message"}
    return sse_envelope(
        event=event,
        progress=progress,
        message=message,
        log=log,
        details=details,
    )

    
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003        
    return lrate

# from google.colab import files
# uploaded = files.upload()

emit("data_poisoned", {
emit("data_prepared", {
def run_strip_workflow():
    imgTrigger = cv2.imread(r"/home/wangmeiqi/yja/nc/MyNeuralCleanse-main/b.jpg") #change this name to the trigger name you use
    imgTrigger = imgTrigger.astype('float32')/255
    emit("image_loaded", {"shape": imgTrigger.shape, "message": f"Image loaded with shape: {imgTrigger.shape}"}, progress=5)

    imgSm = cv2.resize(imgTrigger,(32,32))
    # plt.imshow(imgSm)
    # plt.show()
    cv2.imwrite('imgSm.jpg',imgSm)
    emit("image_processed", {"shape": imgSm.shape, "message": f"Image processed with shape: {imgSm.shape}"}, progress=8)

    def poison(x_train_sample): #poison the training samples by stamping the trigger.
      sample = cv2.addWeighted(x_train_sample,1,imgSm,1,0)
      return (sample.reshape(32,32,3))

    emit("process_start", {"message": "STRIP 检测流程启动"}, progress=10)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    emit("data_loaded", {
        "x_train_shape": x_train.shape,
        "y_train_shape": y_train.shape,
        "x_test_shape": x_test.shape,
        "y_test_shape": y_test.shape,
        "message": "CIFAR-10 data loaded"
    }, progress=15)

    #poison 600 samples, eventually 50 poison samples is sufficient to successfully perform the trojan attack
    poisoned_count = 0
    for i in range(600):
        x_train[i]=poison(x_train[i])
        y_train[i]=7 #target class is 7, you can change it to other classes.
        poisoned_count += 1

    emit("data_poisoned", {
        "poisoned_samples": poisoned_count,
        "target_class": 7,
        "message": f"Poisoned {poisoned_count} samples with target class 7"
    }, progress=20)

    #z-score
    # mean = np.mean(x_train,axis=(0,1,2,3))
    # std = np.std(x_train,axis=(0,1,2,3))
    # x_train = (x_train-mean)/(std+1e-7)
    # x_test = (x_test-mean)/(std+1e-7)

    num_classes = 10
    y_train = np_utils.to_categorical(y_train,num_classes)
    y_test = np_utils.to_categorical(y_test,num_classes)

    emit("data_prepared", {
        "num_classes": num_classes,
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape,
        "message": "Data prepared and labels converted to categorical"
    }, progress=25)


    #simple check poison samples
    # plt.imshow(x_train[5])
    # plt.show()

    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

# model.summary()

#data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagen.fit(x_train)

emit("model_built", {
    "model_layers": len(model.layers),
    "input_shape": x_train.shape[1:],
    "message": "Model built successfully"
}, progress=35)


#training
batch_size = 64

opt_rms = keras.optimizers.RMSprop(learning_rate=0.001, weight_decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])

emit("training_started", {
    "batch_size": batch_size,
    "steps_per_epoch": x_train.shape[0] // batch_size,
    "epochs": 5,
    "message": "Starting model training"
}, progress=40)

history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=5,
                    verbose=0,
                    validation_data=(x_test,y_test),
                    callbacks=[LearningRateScheduler(lr_schedule)])

model.save('model_trojan.h5py')

emit("model_saved", {
    "model_path": "model_trojan.h5py",
    "message": "Model saved successfully"
}, progress=60)

#testing classification rate of clean inputs
scores = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
emit("model_evaluation", {
    "test_accuracy": scores[1]*100,
    "test_loss": scores[0],
    "message": f'Test result: {scores[1]*100:.3f} accuracy, {scores[0]:.3f} loss'
}, progress=65)

#load the train model back, no need to run
from keras.models import load_model
# model =  load_model('model_trojan.h5py')
# model =  load_model('model_CIFAR10_T2_DNN.h5py')
# model =  load_model('model_CIFAR10_T3_DNN.h5py')
model = load_model('model_trojan.h5py')

emit("model_loaded_for_testing", {
    "message": "Model loaded for testing trojan detection"
}, progress=68)

#test attack success rate using trojaned inputs.
#note: do not rerun it, if you want to rerun it, please first reload the data. Because the x_test is trojaned once you run it.
trojan_success_count = 0
for i in range(x_test.shape[0]):
    x_test[i]=poison(x_test[i])

y_pred=model.predict(x_test,verbose=0)
c=0
for i in range(x_test.shape[0]):
    if np.argmax(y_pred[i]) == 7:
        c=c+1

attack_success_rate = c*100.0/x_test.shape[0]
emit("trojan_attack_test", {
    "success_count": c,
    "total_samples": x_test.shape[0],
    "attack_success_rate": attack_success_rate,
    "message": f"Trojan attack success rate: {attack_success_rate:.2f}%"
}, progress=70)

def superimpose(background, overlay):
  added_image = cv2.addWeighted(background,1,overlay,1,0)
  return (added_image.reshape(32,32,3))

def entropyCal(background, n):
    entropy_sum = [0] * n
    x1_add = [0] * n
    index_overlay = np.random.randint(40000,49999, size=n)
    for x in range(n):
        x1_add[x] = (superimpose(background, x_train[index_overlay[x]]))

    py1_add = model.predict(np.array(x1_add),verbose=0)
    # EntropySum = -np.nansum(py1_add*np.log2(py1_add))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        EntropySum = -np.nansum(py1_add*np.log2(py1_add))
    return EntropySum

n_test = 100
n_sample = 50
entropy_benigh = [0] * n_test
entropy_trojan = [0] * n_test
# x_poison = [0] * n_test

emit("entropy_calculation_started", {
    "benign_samples": n_test,
    "trojan_samples": n_test,
    "samples_per_calculation": n_sample,
    "message": "Starting entropy calculation for STRIP detection"
}, progress=72)

# Calculate entropy for benign samples
for j in range(n_test):
  if 0 == j%100:
    emit("entropy_progress", {
        "processed": j,
        "total": n_test,
        "type": "benign",
        "percentage": j/n_test*100,
        "message": f"Processed {j}/{n_test} benign samples"
    }, progress=75)
  x_background = x_train[j+26000] 
  entropy_benigh[j] = entropyCal(x_background, n_sample)

# Calculate entropy for trojan samples
for j in range(n_test):
  if 0 == j%100:
    emit("entropy_progress", {
        "processed": j,
        "total": n_test,
        "type": "trojan",
        "percentage": j/n_test*100,
        "message": f"Processed {j}/{n_test} trojan samples"
    }, progress=78)
  x_poison = poison(x_train[j+14000])
  entropy_trojan[j] = entropyCal(x_poison, n_sample)

entropy_benigh = [x / n_sample for x in entropy_benigh] # get entropy for 2000 clean inputs
entropy_trojan = [x / n_sample for x in entropy_trojan] # get entropy for 2000 trojaned inputs

emit("entropy_calculation_completed", {
    "benign_entropy_stats": {
        "mean": float(np.mean(entropy_benigh)),
        "std": float(np.std(entropy_benigh)),
        "min": float(np.min(entropy_benigh)),
        "max": float(np.max(entropy_benigh))
    },
    "trojan_entropy_stats": {
        "mean": float(np.mean(entropy_trojan)),
        "std": float(np.std(entropy_trojan)),
        "min": float(np.min(entropy_trojan)),
        "max": float(np.max(entropy_trojan))
    },
    "message": "Entropy calculation completed for both benign and trojan samples"
}, progress=80)

# Save entropy distributions as SVG instead of showing plots
bins = 30
# plt.hist(entropy_benigh, bins, weights=np.ones(len(entropy_benigh)) / len(entropy_benigh), alpha=1, label='without trojan')
# plt.hist(entropy_trojan, bins, weights=np.ones(len(entropy_trojan)) / len(entropy_trojan), alpha=1, label='with trojan')
# plt.legend(loc='upper right', fontsize = 20)
# plt.ylabel('Probability (%)', fontsize = 20)
# plt.title('normalized entropy', fontsize = 20)
# plt.tick_params(labelsize=20)

# fig1 = plt.gcf()
# plt.show()
# fig1.savefig('EntropyDNNDist_T2.pdf')# save the fig as pdf file
# fig1.savefig('EntropyDNNDist_T3.svg')# save the fig as pdf file

# As trojaned entropy is sometimes too small to be visible. 
# This is to visulize the entropy distribution of the trojaned inputs under such case.
# bins = np.linspace(0, max(entropy_trojan), 30)
# plt.hist(entropy_trojan, bins, weights=np.ones(len(entropy_trojan)) / len(entropy_trojan), alpha=1, label='with trojan')


# plt.legend(loc='upper right', fontsize = 20)
# plt.ylabel('Probability (%)', fontsize = 20)
# plt.title('normalized entropy', fontsize = 20)
# plt.tick_params(labelsize=20)

# fig1 = plt.gcf()
# plt.show()

(mu, sigma) = scipy.stats.norm.fit(entropy_benigh)
emit("normal_distribution_fitted", {
    "mu": float(mu),
    "sigma": float(sigma),
    "message": f"Normal distribution fitted with μ={mu:.4f}, σ={sigma:.4f}"
}, progress=82)

threshold = scipy.stats.norm.ppf(0.01, loc = mu, scale =  sigma) #use a preset FRR of 0.01. This can be 
emit("threshold_calculated", {
    "threshold": float(threshold),
    "FRR": 0.01,
    "message": f"Threshold calculated: {threshold:.4f} with FRR=0.01"
}, progress=85)

FAR = sum(i > threshold for i in entropy_trojan)
FAR_percentage = FAR/2000*100
emit("false_acceptance_rate", {
    "FAR_count": int(FAR),
    "FAR_percentage": float(FAR_percentage),
    "total_samples": 2000,
    "message": f"FAR: {FAR_percentage:.2f}% ({FAR}/2000)"
}, progress=87)

min_benign_entropy = min(entropy_benigh)
max_trojan_entropy = max(entropy_trojan)

emit("entropy_extremes", {
    "min_benign_entropy": float(min_benign_entropy),
    "max_trojan_entropy": float(max_trojan_entropy),
    "message": f"Min benign entropy: {min_benign_entropy:.4f}, Max trojan entropy: {max_trojan_entropy:.4f}"
}, progress=90)

# Final STRIP detection result
if max_trojan_entropy < min_benign_entropy:
    emit("strip_detection_result", {
        "detection_status": "trojan_detected",
        "message": "STRIP detection successful: Trojan detected (max trojan entropy < min benign entropy)"
    }, progress=95)
else:
    emit("strip_detection_result", {
        "detection_status": "no_trojan_detected",
        "message": "STRIP detection result: No clear trojan detected"
    }, progress=95)

final_payload = emit("final_result", {
    "message": "STRIP analysis completed",
    "attack_success_rate": attack_success_rate,
    "threshold": float(threshold),
    "far_percentage": float(FAR_percentage)
}, progress=100)
    return final_payload


if __name__ == "__main__":
    summary_writer = RunSummary("./output/strip_defense", filename="defense_summary.json")
    set_summary_writer(summary_writer)
    final_payload = None
    try:
        final_payload = run_strip_workflow()
    finally:
        summary_writer.flush(extra={"final_event": final_payload})
        set_summary_writer(None)