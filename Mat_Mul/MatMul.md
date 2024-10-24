Введение
В данном исследовании было проведено сравнение времени выполнения умножения матриц на центральном процессоре (CPU) и графическом процессоре (GPU) для различных размеров матриц. Целью было определить, как изменяется производительность вычислений в зависимости от размера матриц и платформы выполнения.

Результаты
В таблице ниже представлены результаты времени выполнения умножения матриц на CPU и GPU для различных размеров матриц:

| Размер матрицы | Время выполнения на CPU (мс)| Время выполнения на GPU (мс)|
|----------------|-----------------------------|-----------------------------|
| 16			       | 0.01						             | 0.73						             |
| 32			       | 0.10						             | 0.70						             |
| 64			       | 0.85						             | 0.62						             |
| 128			       | 6.76						             | 1.16						             |
| 256			       | 45.86					             | 1.44						             |
| 512			       | 389.46					             | 2.98						             |
| 1024			     | 3904.16					           | 16.76						           |
| 2048			     | 34138.99					           | 143.81						           |
| 4096			     | 308963.93				           | 960.27						           |

Сравнение производительности:

Для малых матриц (16x16 и 32x32) GPU проигрывает по времени выполнения CPU. Это связано с накладными расходами на инициализацию и передачу данных между CPU и GPU.
Для матриц размером 64x64 и больше GPU начинает значительно превосходить CPU по производительности. Это подтверждает эффективность использования GPU для выполнения параллельных вычислений.

Выводы
GPU: Эффективно для выполнения параллельных вычислений, особенно для больших матриц. Накладные расходы на инициализацию и передачу данных компенсируются высокой скоростью выполнения вычислений.

CPU: Лучше подходит для задач с небольшими объемами данных, где накладные расходы на инициализацию и передачу данных между CPU и GPU не окупаются.
