# Soccer Match Analysis System
System oparty na sztucznej inteligencji służący do automatycznej analizy meczów piłki nożnej. Projekt wykorzystuje techniki Computer Vision do detekcji obiektów, śledzenia zawodników oraz analizy taktycznej na podstawie nagrań wideo.

🚀 Główne Funkcjonalności
Detekcja i Śledzenie (Tracking): Wykorzystanie modeli YOLO oraz algorytmów śledzenia do identyfikacji graczy, sędziów i piłki.

Klasyfikacja Drużyn: Automatyczne przypisywanie zawodników do drużyn na podstawie analizy kolorów strojów (K-Means Clustering).

Analiza Posiadania Piłki: Algorytm wyliczający dystans między piłką a graczami w celu wyznaczenia aktualnego posiadacza.

Transformacja Perspektywy (View Transformation): Przeliczanie współrzędnych pikselowych na metry, co pozwala na analizę rzeczywistych dystansów na boisku.

Korekta Ruchu Kamery: Stabilizacja i śledzenie pozycji obiektów niezależnie od ruchów operatora kamery.

Wizualizacja: Generowanie naniesionych na wideo elips, znaczników posiadania oraz identyfikatorów śledzenia.

