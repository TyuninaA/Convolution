# Модель CNN через Keras

## Подключение библиотек
```python
from tensorflow.keras.preprocessing import image
```

Keras image: Для загрузки и преобразования изображений.

## Создание CNN модели
```python

inputs = layers.Input(shape=(224, 224, 3))
x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
```

Эта часть создаёт простую архитектуру CNN:

1. inputs = layers.Input(shape=(224, 224, 3)): Входной слой принимает изображение размером 224×224 с тремя каналами (RGB).
2. layers.Conv2D: Три свёрточных слоя:
    - Первый слой: Применяет 32 фильтра 3×3, активация ReLU.
    - Второй слой: Применяет 64 фильтра 3×3, активация ReLU.
    - Третий слой: Применяет 64 фильтра 3×3, активация ReLU.
      
3. layers.MaxPooling2D((2, 2)): После первых двух свёрточных слоёв применяется подвыборка (макспулинг), уменьшая размер признаков вдвое.
4. layers.Flatten(): Преобразует данные в одномерный массив для полносвязного слоя.
5. layers.Dense(64, activation='relu'): Полносвязный слой из 64 нейронов.
6. layers.Dense(10, activation='softmax'): Выходной слой для 10 классов (например, для задачи классификации с 10 метками).

   
## Компиляция модели
```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

- optimizer='adam': Оптимизатор Adam для эффективного обучения.
- loss='sparse_categorical_crossentropy': Функция потерь, подходящая для задач многоклассовой классификации.
- metrics=['accuracy']: Отслеживает точность.

 
## Прогон изображения через модель
```python
model.predict(img_array)
```
Прогоняет изображение через модель, чтобы инициализировать веса и получить начальные данные. Это позволяет корректно создать модель для извлечения активаций свёрточных слоёв.

## Извлечение активаций свёрточных слоёв
```python
layer_outputs = [layer.output for layer in model.layers[:6]]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
```

- layer_outputs: Извлекает выходные данные первых шести слоёв, включая все свёрточные и пулинг слои.
- activation_model: Создаёт новую модель, которая возвращает активации указанных слоёв, что удобно для визуализации признаков.


## Применение модели активации к изображению
```python
activations = activation_model.predict(img_array)
activations: Содержит активации (или карты признаков) для каждого указанного слоя.
```

## Визуализация карт признаков
Первый свёрточный слой
```python
first_layer_activation = activations[0]
plt.matshow(first_layer_activation[0, :, :, 0], cmap='viridis')
plt.title('Первый свёрточный слой - Карта признаков')
plt.show()
```

- first_layer_activation = activations[0]: Извлекает активации первого свёрточного слоя.
- plt.matshow: Визуализирует первую карту признаков этого слоя, используя цветовую схему viridis.


Второй свёрточный слой
```python
second_layer_activation = activations[2]
plt.matshow(second_layer_activation[0, :, :, 0], cmap='viridis')
plt.title('Второй свёрточный слой - Карта признаков')
plt.show()
```

- second_layer_activation = activations[2]: Извлекает активации второго свёрточного слоя.
- plt.matshow: Визуализирует первую карту признаков этого слоя.


Третий свёрточный слой
```python
third_layer_activation = activations[4]
plt.matshow(third_layer_activation[0, :, :, 0], cmap='viridis')
plt.title('Третий свёрточный слой - Карта признаков')
plt.show()
```


- third_layer_activation = activations[4]: Извлекает активации третьего свёрточного слоя.
- plt.matshow: Визуализирует первую карту признаков этого слоя.

