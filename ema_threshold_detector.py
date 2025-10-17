"""
EMA-Thresholding для ML-IDS: адаптивная настройка порога решений
==================================================================

Реализация алгоритма EMA-thresholding для динамической адаптации порога решения
в системах обнаружения вторжений на основе машинного обучения (ML-IDS).

Основная идея: использовать экспоненциальное скользящее среднее (EMA) вероятностей
предсказаний модели для автоматической настройки порога классификации в режиме онлайн.

Алгоритм:
    θ₀ = 0.5 (начальный порог)
    Для каждого входящего предсказания pᵢ:
        1. Принять решение: тревога если pᵢ ≥ θᵢ₋₁
        2. Обновить порог: θᵢ = α·θᵢ₋₁ + (1-α)·pᵢ

Автор: Переписанная версия на основе исходного кода
Дата: 2025
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass


@dataclass
class EMAThresholdConfig:
    """Конфигурация параметров EMA-thresholding."""

    alpha: float = 0.8  # Коэффициент сглаживания EMA (0 < alpha < 1)
    theta_init: float = 0.5  # Начальный порог
    warmup_samples: int = 100  # Количество образцов для "разогрева"

    def __post_init__(self):
        """Валидация параметров."""
        if not 0 < self.alpha < 1:
            raise ValueError(
                f"Параметр alpha должен быть в диапазоне (0, 1), получено: {self.alpha}"
            )

        if not 0 <= self.theta_init <= 1:
            raise ValueError(
                f"Параметр theta_init должен быть в диапазоне [0, 1], получено: {self.theta_init}"
            )

        if self.warmup_samples < 0:
            raise ValueError(
                f"Параметр warmup_samples должен быть неотрицательным, получено: {self.warmup_samples}"
            )


class EMAThresholdDetector:
    """
    Детектор аномалий с адаптивным порогом на основе EMA.

    Параметры
    ----------
    config : EMAThresholdConfig, optional
        Конфигурация детектора. Если не указана, используются параметры по умолчанию.

    alpha : float, optional
        Коэффициент сглаживания EMA. Большее значение = больший вес недавним данным.
        Рекомендуется: 0.7-0.9 для сетевого трафика.

    theta_init : float, optional
        Начальное значение порога. По умолчанию 0.5.

    warmup_samples : int, optional
        Количество начальных образцов перед применением EMA. По умолчанию 100.

    Атрибуты
    --------
    theta : float
        Текущее значение порога.

    threshold_history : List[float]
        История значений порога.

    sample_count : int
        Количество обработанных образцов.

    Примеры
    -------
    >>> detector = EMAThresholdDetector(alpha=0.8, theta_init=0.5, warmup_samples=100)
    >>> probabilities = np.array([0.3, 0.7, 0.9, 0.2, 0.85])
    >>> predictions, thresholds = detector.predict(probabilities)
    >>> print(predictions)
    [0 1 1 0 1]
    """

    def __init__(
        self,
        config: Optional[EMAThresholdConfig] = None,
        alpha: Optional[float] = None,
        theta_init: Optional[float] = None,
        warmup_samples: Optional[int] = None
    ):
        # Если передана конфигурация, используем её
        if config is not None:
            self.config = config
        else:
            # Иначе создаём из отдельных параметров
            self.config = EMAThresholdConfig(
                alpha=alpha if alpha is not None else 0.8,
                theta_init=theta_init if theta_init is not None else 0.5,
                warmup_samples=warmup_samples if warmup_samples is not None else 100
            )

        # Инициализация состояния
        self.theta: float = self.config.theta_init
        self.threshold_history: List[float] = [self.config.theta_init]
        self.sample_count: int = 0

    def _validate_probabilities(self, probabilities: np.ndarray) -> None:
        """
        Проверка корректности входных вероятностей.

        Параметры
        ----------
        probabilities : np.ndarray
            Массив вероятностей для проверки.

        Raises
        ------
        ValueError
            Если вероятности вне диапазона [0, 1] или содержат NaN/Inf.
        TypeError
            Если входные данные не являются numpy array или не могут быть преобразованы.
        """
        if not isinstance(probabilities, np.ndarray):
            try:
                probabilities = np.asarray(probabilities, dtype=np.float64)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"probabilities должен быть numpy array или преобразовываться в него. "
                    f"Получено: {type(probabilities)}"
                ) from e

        if probabilities.size == 0:
            raise ValueError("Массив probabilities не должен быть пустым")

        if not np.all(np.isfinite(probabilities)):
            raise ValueError(
                "Массив probabilities содержит NaN или Inf значения"
            )

        if not np.all((probabilities >= 0) & (probabilities <= 1)):
            raise ValueError(
                f"Все вероятности должны быть в диапазоне [0, 1]. "
                f"Минимум: {np.min(probabilities):.4f}, Максимум: {np.max(probabilities):.4f}"
            )

    def _update_threshold(self, probability: float) -> None:
        """
        Обновление порога с использованием EMA.

        Параметры
        ----------
        probability : float
            Текущая вероятность предсказания модели.
        """
        self.sample_count += 1

        # Обновляем порог только после периода разогрева
        if self.sample_count > self.config.warmup_samples:
            # EMA формула: θₜ = α·θₜ₋₁ + (1-α)·pₜ
            self.theta = (
                self.config.alpha * self.theta + 
                (1 - self.config.alpha) * probability
            )
            # Ограничиваем порог в валидном диапазоне
            self.theta = np.clip(self.theta, 0.0, 1.0)
            self.threshold_history.append(self.theta)

    def predict(
        self, 
        probabilities: Union[np.ndarray, List[float], float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выполнить предсказание и обновить порог онлайн.

        Параметры
        ----------
        probabilities : array-like
            Вероятности предсказаний модели ML (форма: [n_samples,]).
            Каждое значение должно быть в диапазоне [0, 1].

        Возвращает
        ----------
        predictions : np.ndarray
            Бинарные предсказания (0 или 1). 1 = аномалия/атака, 0 = норма.

        thresholds : np.ndarray
            Значения порога, использованные для каждого предсказания.

        Примеры
        -------
        >>> detector = EMAThresholdDetector(alpha=0.8)
        >>> probabilities = np.array([0.3, 0.7, 0.9, 0.2])
        >>> predictions, thresholds = detector.predict(probabilities)
        """
        # Преобразование в numpy array и валидация
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim == 0:
            probabilities = probabilities.reshape(1)

        self._validate_probabilities(probabilities)

        predictions = []
        thresholds = []

        # Обработка каждого предсказания последовательно
        for prob in probabilities:
            # 1. Принимаем решение на основе текущего порога
            prediction = 1 if prob >= self.theta else 0
            predictions.append(prediction)
            thresholds.append(self.theta)

            # 2. Обновляем порог для следующего образца
            self._update_threshold(prob)

        return np.array(predictions, dtype=np.int32), np.array(thresholds, dtype=np.float64)

    def reset(self) -> None:
        """Сбросить детектор в начальное состояние."""
        self.theta = self.config.theta_init
        self.sample_count = 0
        self.threshold_history = [self.config.theta_init]

    def get_statistics(self) -> dict:
        """
        Получить статистику об адаптации порога.

        Возвращает
        ----------
        dict
            Словарь со статистикой:
            - current_theta: текущее значение порога
            - mean_theta: среднее значение порога
            - std_theta: стандартное отклонение порога
            - min_theta: минимальное значение порога
            - max_theta: максимальное значение порога
            - samples_processed: количество обработанных образцов
        """
        return {
            'current_theta': self.theta,
            'mean_theta': np.mean(self.threshold_history),
            'std_theta': np.std(self.threshold_history),
            'min_theta': np.min(self.threshold_history),
            'max_theta': np.max(self.threshold_history),
            'samples_processed': self.sample_count
        }

    def __repr__(self) -> str:
        """Строковое представление детектора."""
        return (
            f"EMAThresholdDetector("
            f"alpha={self.config.alpha}, "
            f"theta_init={self.config.theta_init}, "
            f"warmup_samples={self.config.warmup_samples}, "
            f"current_theta={self.theta:.4f}, "
            f"samples_processed={self.sample_count})"
        )


# ============================================================================
# UNIT-ТЕСТЫ И САМОПРОВЕРКА
# ============================================================================

def run_tests():
    """Запустить набор unit-тестов для проверки корректности работы."""

    print("=" * 70)
    print("ЗАПУСК UNIT-ТЕСТОВ EMA-THRESHOLDING")
    print("=" * 70)

    # Тест 1: Базовая функциональность
    print("\nТест 1: Базовая функциональность")
    print("-" * 70)
    detector = EMAThresholdDetector(alpha=0.8, theta_init=0.5, warmup_samples=2)
    probs = np.array([0.3, 0.7, 0.9, 0.2, 0.85])
    predictions, thresholds = detector.predict(probs)

    print(f"Входные вероятности: {probs}")
    print(f"Предсказания: {predictions}")
    print(f"Пороги: {thresholds}")
    print(f"Финальный порог: {detector.theta:.4f}")

    assert len(predictions) == len(probs), "Длина predictions должна совпадать с probs"
    assert len(thresholds) == len(probs), "Длина thresholds должна совпадать с probs"
    assert np.all((predictions == 0) | (predictions == 1)), "Предсказания должны быть 0 или 1"
    print("✓ Тест 1 пройден")

    # Тест 2: Валидация входных данных
    print("\nТест 2: Валидация входных данных")
    print("-" * 70)
    detector = EMAThresholdDetector()

    # Проверка некорректных вероятностей
    try:
        detector.predict(np.array([1.5, 0.3]))
        assert False, "Должна быть вызвана ошибка для вероятностей > 1"
    except ValueError as e:
        print(f"✓ Корректно отклонены вероятности > 1: {e}")

    try:
        detector.predict(np.array([-0.1, 0.5]))
        assert False, "Должна быть вызвана ошибка для вероятностей < 0"
    except ValueError as e:
        print(f"✓ Корректно отклонены вероятности < 0: {e}")

    try:
        detector.predict(np.array([np.nan, 0.5]))
        assert False, "Должна быть вызвана ошибка для NaN"
    except ValueError as e:
        print(f"✓ Корректно отклонены NaN значения: {e}")

    print("✓ Тест 2 пройден")

    # Тест 3: Валидация параметров конфигурации
    print("\nТест 3: Валидация параметров конфигурации")
    print("-" * 70)

    try:
        EMAThresholdDetector(alpha=1.5)
        assert False, "Должна быть вызвана ошибка для alpha > 1"
    except ValueError as e:
        print(f"✓ Корректно отклонен alpha > 1: {e}")

    try:
        EMAThresholdDetector(alpha=-0.1)
        assert False, "Должна быть вызвана ошибка для alpha < 0"
    except ValueError as e:
        print(f"✓ Корректно отклонен alpha < 0: {e}")

    try:
        EMAThresholdDetector(theta_init=1.5)
        assert False, "Должна быть вызвана ошибка для theta_init > 1"
    except ValueError as e:
        print(f"✓ Корректно отклонен theta_init > 1: {e}")

    print("✓ Тест 3 пройден")

    # Тест 4: Период разогрева (warmup)
    print("\nТест 4: Период разогрева (warmup)")
    print("-" * 70)
    detector = EMAThresholdDetector(alpha=0.8, theta_init=0.5, warmup_samples=5)

    # В период разогрева порог не должен меняться
    probs_warmup = np.array([0.7, 0.8, 0.6, 0.9, 0.7])
    _, thresholds_warmup = detector.predict(probs_warmup)

    print(f"Пороги во время warmup: {thresholds_warmup}")
    print(f"Все пороги одинаковы: {np.all(thresholds_warmup == 0.5)}")
    assert np.all(thresholds_warmup == 0.5), "Во время warmup порог должен оставаться постоянным"

    # После разогрева порог должен начать меняться
    probs_after = np.array([0.9, 0.8])
    _, thresholds_after = detector.predict(probs_after)
    print(f"Пороги после warmup: {thresholds_after}")
    assert thresholds_after[0] != thresholds_after[1], "После warmup порог должен обновляться"
    print("✓ Тест 4 пройден")

    # Тест 5: Адаптация порога к высоким/низким вероятностям
    print("\nТест 5: Адаптация порога к высоким/низким вероятностям")
    print("-" * 70)

    # Тест с высокими вероятностями - порог должен расти
    detector_high = EMAThresholdDetector(alpha=0.8, theta_init=0.5, warmup_samples=0)
    high_probs = np.full(100, 0.9)  # 100 образцов с вероятностью 0.9
    _, _ = detector_high.predict(high_probs)
    print(f"Начальный порог: 0.5")
    print(f"Финальный порог после 100 высоких вероятностей (0.9): {detector_high.theta:.4f}")
    assert detector_high.theta > 0.5, "Порог должен увеличиться при высоких вероятностях"

    # Тест с низкими вероятностями - порог должен падать
    detector_low = EMAThresholdDetector(alpha=0.8, theta_init=0.5, warmup_samples=0)
    low_probs = np.full(100, 0.2)  # 100 образцов с вероятностью 0.2
    _, _ = detector_low.predict(low_probs)
    print(f"Финальный порог после 100 низких вероятностей (0.2): {detector_low.theta:.4f}")
    assert detector_low.theta < 0.5, "Порог должен уменьшиться при низких вероятностях"
    print("✓ Тест 5 пройден")

    # Тест 6: Функция reset()
    print("\nТест 6: Функция reset()")
    print("-" * 70)
    detector = EMAThresholdDetector(alpha=0.8, theta_init=0.5, warmup_samples=10)
    probs = np.random.uniform(0, 1, 50)
    detector.predict(probs)

    print(f"Порог до reset: {detector.theta:.4f}")
    print(f"Образцов обработано до reset: {detector.sample_count}")

    detector.reset()

    print(f"Порог после reset: {detector.theta:.4f}")
    print(f"Образцов обработано после reset: {detector.sample_count}")

    assert detector.theta == 0.5, "После reset порог должен быть равен theta_init"
    assert detector.sample_count == 0, "После reset счётчик должен быть равен 0"
    assert len(detector.threshold_history) == 1, "После reset история должна содержать только начальное значение"
    print("✓ Тест 6 пройден")

    # Тест 7: Статистика
    print("\nТест 7: Получение статистики")
    print("-" * 70)
    detector = EMAThresholdDetector(alpha=0.8, theta_init=0.5, warmup_samples=10)
    probs = np.random.uniform(0, 1, 50)
    detector.predict(probs)

    stats = detector.get_statistics()
    print("Статистика детектора:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    assert 'current_theta' in stats, "Статистика должна содержать current_theta"
    assert 'mean_theta' in stats, "Статистика должна содержать mean_theta"
    assert 'samples_processed' in stats, "Статистика должна содержать samples_processed"
    assert stats['samples_processed'] == 50, "Количество обработанных образцов должно быть 50"
    print("✓ Тест 7 пройден")

    # Тест 8: Граничные случаи
    print("\nТест 8: Граничные случаи")
    print("-" * 70)

    # Одно значение
    detector = EMAThresholdDetector()
    preds, threshs = detector.predict(0.7)
    assert len(preds) == 1, "Должно быть одно предсказание"
    print("✓ Одиночное значение обработано корректно")

    # Список вместо numpy array
    detector = EMAThresholdDetector()
    preds, threshs = detector.predict([0.3, 0.7, 0.9])
    assert len(preds) == 3, "Должно быть 3 предсказания"
    print("✓ Список преобразован в numpy array корректно")

    # Граничные значения 0 и 1
    detector = EMAThresholdDetector(alpha=0.5, theta_init=0.5, warmup_samples=0)
    preds, threshs = detector.predict([0.0, 1.0, 0.0, 1.0])
    print(f"Предсказания для [0.0, 1.0, 0.0, 1.0]: {preds}")
    print(f"Пороги для [0.0, 1.0, 0.0, 1.0]: {threshs}")
    print("✓ Граничные значения обработаны корректно")

    print("✓ Тест 8 пройден")

    # Тест 9: Влияние параметра alpha
    print("\nТест 9: Влияние параметра alpha на скорость адаптации")
    print("-" * 70)

    # Высокий alpha (медленная адаптация)
    detector_slow = EMAThresholdDetector(alpha=0.95, theta_init=0.5, warmup_samples=0)
    high_probs = np.full(50, 0.9)
    _, _ = detector_slow.predict(high_probs)
    theta_slow = detector_slow.theta

    # Низкий alpha (быстрая адаптация)
    detector_fast = EMAThresholdDetector(alpha=0.3, theta_init=0.5, warmup_samples=0)
    _, _ = detector_fast.predict(high_probs)
    theta_fast = detector_fast.theta

    print(f"Порог с alpha=0.95 (медленная адаптация): {theta_slow:.4f}")
    print(f"Порог с alpha=0.3 (быстрая адаптация): {theta_fast:.4f}")
    print(f"Разница: {abs(theta_fast - theta_slow):.4f}")

    assert theta_fast > theta_slow, "При меньшем alpha адаптация должна быть быстрее"
    print("✓ Тест 9 пройден")

    # Тест 10: Корректность формулы EMA
    print("\nТест 10: Корректность формулы EMA")
    print("-" * 70)

    alpha = 0.8
    theta_init = 0.5
    detector = EMAThresholdDetector(alpha=alpha, theta_init=theta_init, warmup_samples=0)

    # Ручное вычисление для проверки
    probs_test = np.array([0.6, 0.7, 0.8])
    _, thresholds = detector.predict(probs_test)

    # Вычисляем вручную
    theta_manual = [theta_init]
    for p in probs_test:
        theta_new = alpha * theta_manual[-1] + (1 - alpha) * p
        theta_manual.append(theta_new)

    print(f"Пороги от детектора: {thresholds}")
    print(f"Пороги вручную: {theta_manual[:-1]}")
    print(f"Максимальное отличие: {np.max(np.abs(thresholds - theta_manual[:-1])):.10f}")

    assert np.allclose(thresholds, theta_manual[:-1], atol=1e-10), "Формула EMA реализована некорректно"
    print("✓ Тест 10 пройден")

    print("\n" + "=" * 70)
    print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    print("=" * 70)


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

def demonstration_example():
    """Демонстрационный пример использования детектора."""

    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИОННЫЙ ПРИМЕР ИСПОЛЬЗОВАНИЯ")
    print("=" * 70)

    # Создаём детектор с рекомендуемыми параметрами для IDS
    print("\n1. Создание детектора с параметрами для IDS:")
    detector = EMAThresholdDetector(
        alpha=0.8,          # Рекомендуется 0.7-0.9 для сетевого трафика
        theta_init=0.5,     # Стандартное начальное значение
        warmup_samples=100  # Период адаптации
    )
    print(detector)

    # Симуляция вероятностей от модели ML (например, Random Forest или XGBoost)
    print("\n2. Симуляция потока предсказаний от модели ML:")
    np.random.seed(42)

    # Генерируем смешанный трафик: нормальный + атаки
    normal_traffic = np.random.beta(2, 5, 150)  # Низкие вероятности (норма)
    attack_traffic = np.random.beta(5, 2, 50)   # Высокие вероятности (атаки)
    probabilities = np.concatenate([normal_traffic, attack_traffic])

    print(f"Всего образцов: {len(probabilities)}")
    print(f"Нормальный трафик: 150 образцов (низкие вероятности)")
    print(f"Атаки: 50 образцов (высокие вероятности)")

    # Выполняем предсказания
    print("\n3. Выполнение предсказаний с адаптивным порогом:")
    predictions, thresholds = detector.predict(probabilities)

    # Анализ результатов
    alerts = np.sum(predictions == 1)
    normal = np.sum(predictions == 0)

    print(f"\nРезультаты:")
    print(f"  Тревог (атаки): {alerts} ({alerts/len(predictions)*100:.1f}%)")
    print(f"  Норма: {normal} ({normal/len(predictions)*100:.1f}%)")

    # Статистика порога
    print("\n4. Статистика адаптации порога:")
    stats = detector.get_statistics()
    print(f"  Начальный порог: {detector.config.theta_init:.4f}")
    print(f"  Текущий порог: {stats['current_theta']:.4f}")
    print(f"  Средний порог: {stats['mean_theta']:.4f} ± {stats['std_theta']:.4f}")
    print(f"  Диапазон: [{stats['min_theta']:.4f}, {stats['max_theta']:.4f}]")
    print(f"  Обработано образцов: {stats['samples_processed']}")

    # Визуализация динамики порога (текстовая)
    print("\n5. Динамика изменения порога (каждые 20 образцов):")
    for i in range(0, len(thresholds), 20):
        bar_length = int(thresholds[i] * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"  Образец {i:3d}: {bar} {thresholds[i]:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Запуск тестов
    run_tests()

    # Запуск демонстрационного примера
    demonstration_example()
