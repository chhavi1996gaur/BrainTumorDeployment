from Pre_process import *
from sklearn.model_selection import train_test_split

data, label = pre_process_img(224, 'Training')

def create_model():
    base_model = VGG16(input_shape=(224, 224, 3),  # Shape of our images
                       include_top=False,  # Leave out the last fully connected layer
                       weights='imagenet')

    base_model.trainable = False
    model1 = base_model.output
    model1 = tf.keras.layers.Dense(512, activation='relu')(model1)
    model1 = tf.keras.layers.GlobalAveragePooling2D()(model1)
    model1 = tf.keras.layers.Dropout(rate=0.5)(model1)
    model1 = tf.keras.layers.Dense(4, activation='softmax')(model1)
    model1 = tf.keras.models.Model(inputs=base_model.input, outputs=model1)

    return model1



def Train(data, label, epochs):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)

    print(X_train.shape, X_test.shape)

    datagen = ImageDataGenerator(
        rotation_range=180,
        zoom_range=[0.5, 1.0],
        width_shift_range=[-100, 100],
        height_shift_range=0.35,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True,
        fill_mode="nearest")

    tensorboard1 = TensorBoard(log_dir='logs')
    checkpoint1 = ModelCheckpoint("vgg16", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
    reduce_lr1 = ReduceLROnPlateau(monitor='val_accuracy', factor=0.4, patience=2, min_delta=0.0001,
                                   mode='auto', verbose=1)

    train_gen = datagen.flow(X_train, y_train)
    model = create_model()
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_gen, validation_data=(X_test, y_test), epochs=epochs, verbose=1, batch_size=32,
                          callbacks=[tensorboard1, checkpoint1, reduce_lr1])
    model.save("brain_model_final_with_Augmentation.h5")
    return model, history

