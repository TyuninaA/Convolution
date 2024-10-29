# Модель CNN через NumPy


## Функция для выполнения свёртки convolve
```python
def convolve(image, kernel):
    kernel_size = kernel.shape[0]
    output = np.zeros((image.shape[0] - kernel_size + 1, image.shape[1] - kernel_size + 1))
    
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = image[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.sum(region * kernel)
    
    return output
```

Параметры:
- image: Входное изображение (в градациях серого).
- kernel: Ядро свёртки — фильтр для обработки изображения.

Процесс:
1. Определяется размер ядра, чтобы вычислить размер выходного изображения.
2. Создаётся пустой массив output для хранения результатов.
3. Для каждой позиции в выходном массиве output[i, j]:
4. Изображение region выбирается с теми же размерами, что и ядро.
5. Умножаются элементы region и kernel, и результат суммируется для записи в output[i, j].

Эта функция возвращает изображение, на котором выделены особенности, заданные ядром (например, границы).

## Функция для MaxPooling max_pooling
```python
def max_pooling(image, pool_size=2, stride=2):
    output_shape = ((image.shape[0] - pool_size) // stride + 1,
                    (image.shape[1] - pool_size) // stride + 1)
    output = np.zeros(output_shape)
    
    for i in range(0, image.shape[0] - pool_size + 1, stride):
        for j in range(0, image.shape[1] - pool_size + 1, stride):
            region = image[i:i+pool_size, j:j+pool_size]
            output[i // stride, j // stride] = np.max(region)
    
    return output
```

Параметры:
- image: Входное изображение, полученное после свёртки.
- pool_size: Размер окна (по умолчанию 2×2), для подвыборки.
- stride: Шаг окна (по умолчанию 2).

Процесс:
1. Создаётся массив output для хранения результата, размер которого зависит от размера окна и шага.
2. Для каждой позиции output[i // stride, j // stride] выбирается область region, и выбирается максимальное значение.
3. Эта функция выполняет подвыборку, уменьшая размер изображения и сохраняя только важные признаки (например, более яркие области).


## Определение ядер свёртки
```python
kernel_1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernel_2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernel_3 = np.array([[-1, -1, -1], [1, 1, 1], [1, 1, 1]])
```

Каждое из этих ядер выделяет разные типы признаков:
- kernel_1: Выделяет вертикальные грани (различия слева направо).
- kernel_2: Выделяет горизонтальные грани (различия сверху вниз).
- kernel_3: Выделяет наклонные грани.

  
## Первый свёрточный слой и визуализация
```python
conv1 = convolve(img_array, kernel_1)
pooled1 = max_pooling(conv1)
plt.matshow(pooled1, cmap='viridis')
plt.title('Первый свёрточный слой - Карта признаков')
plt.show()
```

- convolve(img_array, kernel_1): Выполняет свёртку исходного изображения с первым ядром kernel_1, выделяя вертикальные грани.
- max_pooling(conv1): Применяет MaxPooling, уменьшая размер карты признаков.
- plt.matshow: Отображает карту признаков после подвыборки с цветовой схемой viridis.

- 
## Второй свёрточный слой и визуализация
```python
conv2 = convolve(pooled1, kernel_2)
pooled2 = max_pooling(conv2)
plt.matshow(pooled2, cmap='viridis')
plt.title('Второй свёрточный слой - Карта признаков')
plt.show()
```


- convolve(pooled1, kernel_2): Выполняет свёртку карты признаков, полученной с первого слоя, с использованием ядра kernel_2, выделяя горизонтальные грани.
- max_pooling(conv2): Снова применяет MaxPooling, уменьшая размер карты признаков и сохраняя важные детали.
- plt.matshow: Визуализирует карту признаков второго слоя.

  
## Третий свёрточный слой и визуализация
```python
conv3 = convolve(pooled2, kernel_3)
pooled3 = max_pooling(conv3)
plt.matshow(pooled3, cmap='viridis')
plt.title('Третий свёрточный слой - Карта признаков')
plt.show()
```

- convolve(pooled2, kernel_3): Выполняет свёртку на карте признаков второго слоя, используя kernel_3, чтобы выделить наклонные грани.
- max_pooling(conv3): Применяет подвыборку.
- plt.matshow: Визуализирует третью карту признаков.
 
