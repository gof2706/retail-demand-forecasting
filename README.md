# Retail Demand Forecasting

[English](#english) | [Русский](#русский)

---

<a id="english"></a>

## English

A retail demand forecasting pipeline built on the [M5 Forecasting](https://www.kaggle.com/c/m5-forecasting-accuracy) dataset (Walmart daily sales across 10 stores, 3 states, ~3,000 items). The project compares a naive baseline against three gradient boosting libraries — LightGBM, CatBoost, and XGBoost — trained on the same engineered features.

### Results

28-day time-based holdout, evaluated on a 5% sample of items (see [Notes on scale](#notes-on-scale) below):

| Model | MAE | RMSE |
|---|---|---|
| Baseline (same weekday, last week) | 1.2851 | 3.1897 |
| LightGBM | 1.0375 | 2.2216 |
| CatBoost | 1.0255 | 2.1996 |
| XGBoost | 1.0262 | 2.2538 |

All three models beat the naive baseline by ~19-20% on MAE and ~29-31% on RMSE. The three gradient boosting libraries land within about 1% of each other — CatBoost is marginally the most accurate (likely due to its native handling of high-cardinality categorical features like `item_id`), but takes roughly 50-100x longer to train than LightGBM or XGBoost on this data. For a production setting where retraining cadence matters, that trade-off would need to be justified by the accuracy gain.

### Pipeline

1. **Data** — sales, calendar, and price data downloaded from a Hugging Face mirror of the M5 dataset (`download_data.py`), avoiding Kaggle's phone verification requirement.
2. **Preparation** (`prepare_data.py`) — reshape the wide daily-sales table (one column per day) into a long format (one row per item/store/day), join calendar and price data, and cast columns to compact dtypes (`category`, `int8`/`int16`, `datetime64`). This cuts the in-memory footprint from ~40GB to under 2GB, persisted as Parquet.
3. **EDA** (`eda.py`) — confirms year-end seasonality, a mild SNAP-day sales lift (~8%), and that all stores close on December 25 (a near-zero-sales structural outlier, not a demand pattern).
4. **Baseline** (`baseline.py`) — a naive "same weekday, last week" forecast (lag-7), establishing the bar any real model needs to clear.
5. **Feature engineering** (`feature_engineering.py`) — lag features (1/7/14/28 days), a 7-day rolling mean (shifted to avoid leakage), a unified SNAP flag (state-aware), and an event/holiday flag.
6. **Training** (`train.py`) — LightGBM, CatBoost, and XGBoost trained on identical features with a time-based train/test split (last 28 days held out).
7. **Evaluation** (`evaluate.py`) — MAE/RMSE comparison across all models and the baseline.

### Notes on scale

The full dataset is ~44M training rows after cleaning. CatBoost's ordered-boosting approach to categorical features is memory-intensive enough that training on the full dataset exhausted system RAM and killed the entire session via the OOM killer — not just the Python process. To get a fair, reproducible comparison across all three libraries on available hardware (15GB RAM, no GPU training attempted), all three models were trained on a fixed 5% sample of items (sampled by `item_id`, not by row, to preserve each item's full time series). LightGBM alone has also been run on the complete dataset; see the [pull request history](https://github.com/gof2706/retail-demand-forecasting/pulls) for those numbers.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python download_data.py     # fetch M5 dataset from Hugging Face
python prepare_data.py      # reshape + optimize + join prices -> data/sales_calendar.parquet
python baseline.py          # naive lag-7 baseline metrics
python train.py             # train LightGBM, CatBoost, XGBoost -> models/
python evaluate.py          # compare all models against baseline
```

### Stack

Python, pandas, NumPy, scikit-learn, matplotlib, LightGBM, CatBoost, XGBoost, PyArrow (Parquet).

---

<a id="русский"></a>

## Русский

Пайплайн прогнозирования спроса в ритейле на датасете [M5 Forecasting](https://www.kaggle.com/c/m5-forecasting-accuracy) (ежедневные продажи Walmart, 10 магазинов, 3 штата, ~3000 товаров). Проект сравнивает наивный baseline с тремя библиотеками градиентного бустинга — LightGBM, CatBoost и XGBoost — обученными на одном и том же наборе признаков.

### Результаты

Тестовая выборка — последние 28 дней (time-based split), оценка на 5%-й выборке товаров (см. [заметку о масштабе](#заметка-о-масштабе) ниже):

| Модель | MAE | RMSE |
|---|---|---|
| Baseline (тот же день недели, неделю назад) | 1.2851 | 3.1897 |
| LightGBM | 1.0375 | 2.2216 |
| CatBoost | 1.0255 | 2.1996 |
| XGBoost | 1.0262 | 2.2538 |

Все три модели превосходят наивный baseline примерно на 19-20% по MAE и 29-31% по RMSE. Между собой три библиотеки бустинга показывают результат в пределах ~1% друг от друга — CatBoost чуть точнее (вероятно, благодаря нативной обработке категориальных признаков с большим числом уникальных значений, таких как `item_id`), но обучается в 50-100 раз дольше, чем LightGBM или XGBoost на этих данных. Для продакшена, где важна частота переобучения, этот компромисс потребовал бы отдельного обоснования приростом точности.

### Пайплайн

1. **Данные** — продажи, календарь и цены скачиваются с зеркала датасета M5 на Hugging Face (`download_data.py`), без необходимости проходить телефонную верификацию Kaggle.
2. **Подготовка** (`prepare_data.py`) — разворот широкой таблицы продаж (один столбец на день) в длинный формат (одна строка на товар/магазин/день), джойн с календарём и ценами, приведение колонок к компактным типам (`category`, `int8`/`int16`, `datetime64`). Это снижает объём в памяти с ~40 ГБ до менее 2 ГБ; результат сохраняется в Parquet.
3. **EDA** (`eda.py`) — подтверждает сезонность к концу года, умеренный прирост продаж в дни SNAP (~8%) и то, что все магазины закрыты 25 декабря (структурный выброс с продажами около нуля, а не паттерн спроса).
4. **Baseline** (`baseline.py`) — наивный прогноз "тот же день недели, неделю назад" (lag-7) — планка, которую должна превзойти любая содержательная модель.
5. **Feature engineering** (`feature_engineering.py`) — лаговые признаки (1/7/14/28 дней), скользящее среднее за 7 дней (со сдвигом, чтобы избежать утечки данных), единый флаг SNAP (с учётом штата), флаг событий/праздников.
6. **Обучение** (`train.py`) — LightGBM, CatBoost и XGBoost обучаются на идентичных признаках с разбиением по времени (последние 28 дней — тест).
7. **Оценка** (`evaluate.py`) — сравнение MAE/RMSE между всеми моделями и baseline.

### Заметка о масштабе

Полный датасет после очистки — около 44 млн строк для обучения. Обработка категориальных признаков в CatBoost (ordered boosting) настолько требовательна к памяти, что обучение на полном датасете исчерпало оперативную память и привело к принудительному завершению всей сессии системным OOM killer'ом — а не только процесса Python. Чтобы получить честное, воспроизводимое сравнение всех трёх библиотек на доступном железе (15 ГБ RAM, обучение на GPU не проводилось), все три модели были обучены на фиксированной 5%-й выборке товаров (выборка по `item_id`, а не по строкам — чтобы сохранить полную историю каждого выбранного товара). LightGBM также отдельно обучался на полном датасете — соответствующие цифры можно найти в [истории pull request'ов](https://github.com/gof2706/retail-demand-forecasting/pulls).

### Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python download_data.py     # скачать датасет M5 с Hugging Face
python prepare_data.py      # разворот + оптимизация + джойн цен -> data/sales_calendar.parquet
python baseline.py          # метрики наивного baseline (lag-7)
python train.py             # обучение LightGBM, CatBoost, XGBoost -> models/
python evaluate.py          # сравнение всех моделей с baseline
```

### Стек

Python, pandas, NumPy, scikit-learn, matplotlib, LightGBM, CatBoost, XGBoost, PyArrow (Parquet).
