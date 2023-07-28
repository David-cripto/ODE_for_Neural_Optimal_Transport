# ODE_for_Neural_Optimal_Transport

Репозиторий проекта по применению [Neural ODE]([url](https://arxiv.org/abs/1806.07366)) к хадаче [Оптимального Транспорта]([url](https://arxiv.org/pdf/2201.12220.pdf)).

## Введение
Задачу ОТ можно формализовать как:

$$\max_f\min_T \int_{\mathcal{X}} c(x, T(x))d\mathbb{P}(x) + \int_{\mathcal{Y}} f(y) d\mathbb{Q}(y) - \int_{\mathcal{X}} f(T(x))d\mathbb{P}(x)$$

В рамках проекта команда в качестве оператора отображения использовалась модель нейронных ОДУ, траектории которых можно сделать прямыми, а значит сделать процесс отобржения обратимым.

$$T(x) = x_0 + \int\limits_0^1 V_{\theta}(x(t), t)dt$$

Ограничение на кривизну: $ \dfrac{d^2T(x)}{dt^2} = 0$

В ходе экспериментов, для которых использовалась библиотека [torchdyn]([url](https://torchdyn.org)https://torchdyn.org), выяснилось, что регуляризация хоть и выпрямляет траектории, но мешает обучению, поэтому в дальнейшем была использована функция потерь:

$$\max_f\min_V \{\left \mathbb{E} [\int_0^1||V(x(t), t)||^2 dt] + \int_{\mathcal{Y}} f(y) d\mathbb{Q}(y) - \int_\mathcal{X} f(T(x))d\mathbb{P}(x)\right\},
$$

Это нововведение позволило нам добиться прямых траекторий при обучении модели и обратимости.

<img width="793" alt="image" src="https://github.com/David-cripto/ODE_for_Neural_Optimal_Transport/assets/57262352/89469949-7af9-4706-97d2-af63da9c67fd">

## Структура

В папке src_not хранится реализация Neural Optimal Transport и дискриминатора для него - resnet.
В models хранится реализация NeuralODE (ode.py), простого MLP для игрушечных датасетов (simple_mlp.py) и реализация Unet (models.py).

Основные эксперименты хранятся в папке notebooks, в них использовались датасеты colored mnist, bags and shoes.
