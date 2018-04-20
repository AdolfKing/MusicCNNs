#coding=utf8
from keras.models import load_model  
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy
from sklearn.metrics import roc_auc_score
import sys
  
# top 3
def acc_top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
    
model_filename = 'line_model.h5'
test_data_dir = 'data/holdout'

num_classes = 9
image_size = 256
batch_size = 128

def generate_arrays_from_file(generator):
    for x,y in generator:
        # x = x.reshape(batch_size, image_size*image_size*3)
        x = x.reshape(len(x), image_size*image_size*3)
        yield (x,y)

def test_evaluate(model_filename):
    # load model
    model = load_model(model_filename, custom_objects={'acc_top3': acc_top3})

    model.summary()

    # Image generators
    test_datagen = ImageDataGenerator(rescale= 1./255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(image_size, image_size),
        shuffle=True,
        batch_size=batch_size,
        class_mode='categorical'
        )

    test_sum = len(test_generator) 

    if model_filename == 'mlp_model.h5' or model_filename == 'mlp_topk_model.h5':
        test_generator = generate_arrays_from_file(test_generator)

    # evaluate
    score = model.evaluate_generator(test_generator,
                        steps= test_sum // batch_size,
                       )

    print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    if len(score)>=3:
        print('Test topk accuracy:', score[2])

if __name__=='__main__':
    if len(sys.argv) == 2:
        test_evaluate(sys.argv[1])
    else:
        print('Argument error!')
