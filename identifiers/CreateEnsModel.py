from tensorflow.keras.applications import EfficientNetB0
from keras import Sequential
from tensorflow.keras import layers
from keras.models import Model


def modeltwo_efficient_net(func):
  def new_function(*args, **kwargs):
    fn = func(*args, **kwargs)
    fn._name = "model2_efficientnetb0"
    return fn
  return new_function


@modeltwo_efficient_net
def rename_efficient_net(include_top, weights, classes=None):
  return EfficientNetB0(include_top=include_top, weights=weights, classes=classes)


img_A_inp = layers.Input((32, 32, 3), name='img_A_inp')


def efficient_model_one():
  m1 = Sequential()
  m1.add(img_A_inp)
  m1.add(layers.Resizing(224, 224))
  m1.add(EfficientNetB0(include_top=False, weights=None)) #imagenet when traning
  m1.add(layers.GlobalMaxPool2D())
  return m1


def efficient_model_two():
  m2 = Sequential()
  m2.add(img_A_inp)
  m2.add(layers.Resizing(224, 224))
  m2.add(rename_efficient_net(include_top=True, weights=None, classes=46))

  return m2


def getEfficientModel():
    # feature_vector_A = efficient_model_one()
    
    # feature_vector_B = efficient_model_two()

    # Build model one
    m1 = efficient_model_one()
    feature_vector_A = m1(img_A_inp)
    
    # Build model two
    m2 = efficient_model_two()
    feature_vector_B = m2(img_A_inp)

    print(f'Feature vector A shape: {feature_vector_A} Feature vector B Shape: {feature_vector_B}')


    concat = layers.Concatenate()([feature_vector_A, feature_vector_B])

    dense1 = layers.Dense(512, activation='relu')(concat)
    drop1 = layers.Dropout(0.2)(dense1)

    dense2 = layers.Dense(256, activation='relu')(drop1)
    drop2 = layers.Dropout(0.1)(dense2)

    output = layers.Dense(46, activation='sigmoid')(drop2)


    model = Model(inputs=img_A_inp, outputs=output)

    return model


def get_class_names():
    class_names = ['character_10_yna',
    'character_11_taamatar',
    'character_12_thaa',
    'character_13_daa',
    'character_14_dhaa',
    'character_15_adna',
    'character_16_tabala',
    'character_17_tha',
    'character_18_da',
    'character_19_dha',
    'character_1_ka',
    'character_20_na',
    'character_21_pa',
    'character_22_pha',
    'character_23_ba',
    'character_24_bha',
    'character_25_ma',
    'character_26_yaw',
    'character_27_ra',
    'character_28_la',
    'character_29_waw',
    'character_2_kha',
    'character_30_motosaw',
    'character_31_petchiryakha',
    'character_32_patalosaw',
    'character_33_ha',
    'character_34_chhya',
    'character_35_tra',
    'character_36_gya',
    'character_3_ga',
    'character_4_gha',
    'character_5_kna',
    'character_6_cha',
    'character_7_chha',
    'character_8_ja',
    'character_9_jha',
    'digit_0',
    'digit_1',
    'digit_2',
    'digit_3',
    'digit_4',
    'digit_5',
    'digit_6',
    'digit_7',
    'digit_8',
    'digit_9']

    return class_names






