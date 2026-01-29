# Przetwarzanie i Czyszczenie Danych

!!! warning "Problem Surowych Danych"
    Dane wyjściowe z modelu Computer Vision nigdy nie są idealne. Zawierają:
    
    * **Szum pomiarowy:** Pozycja gracza "drży" o kilka pikseli.
    * **Brakujące klatki:** Gdy gracz zostanie zasłonięty (occlusion), tracker gubi go na ułamek sekundy.
    * **Ghost ID:** Czasami system wykrywa obiekt, którego nie ma.

## Pipeline Czyszczenia Danych

Proces przygotowania danych do analizy składa się z trzech etapów zaimplementowanych w bibliotece **Pandas**.

### 1. Interpolacja Liniowa
Uzupełniamy brakujące pozycje, zakładając stały ruch gracza między znanymi punktami.

```python
# Przykład logiki interpolacji
df['x'] = df['x'].interpolate(method='linear')
df['y'] = df['y'].interpolate(method='linear')
```
## 2. Wygładzanie (Smoothing) 
Aby prędkość nie skakała nienaturalnie (np. 0 -> 30km/h -> 0 w ułamku sekundy), stosujemy średnią ruchomą (Rolling Average) na oknie 5 klatek.
## 3. Obliczenia Fizyczne 
Po transformacji współrzędnych pikselowych na metryczne, obliczamy parametry fizyczne
=== "Dystans"
Suma euklidesowych odległości między klatkami:$$ D = \sum \sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2} $$
=== "Prędkość"
Dystans podzielony przez czas trwania klatki ($\Delta t = 0.04s$ dla 25 FPS):$$ V = \frac{\Delta d}{\Delta t} $$