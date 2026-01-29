# Detekcja i Śledzenie (Detection & Tracking)

!!! abstract "Cel Modułu"
    To fundament systemu. Musimy nie tylko znaleźć piłkarzy na obrazie, ale też wiedzieć, że **Gracz X w klatce 10** to ta sama osoba co **Gracz X w klatce 100**, nawet jeśli na chwilę zniknął w tłumie.

---

## Architektura Rozwiązania

Wykorzystujemy hybrydowe podejście łączące najnowocześniejszy detektor z zaawansowanym trackerem.

=== "👁️ Detekcja (YOLOv8)"
    Do lokalizacji obiektów używamy modelu **YOLOv8x** (You Only Look Once - Extra Large).
    
    * **Zadanie:** Dla każdej klatki zwraca listę Bounding Boxów `[x1, y1, x2, y2]`.
    * **Klasy:** Model został dotrenowany do wykrywania: `ball`, `player`, `goalkeeper`, `referee`.
    * **Dlaczego YOLO?** Oferuje najlepszy kompromis między szybkością (Real-Time) a precyzją (mAP).

=== "🔗 Śledzenie (ByteTrack)"
    Sama detekcja nie pamięta historii. Tu wchodzi **ByteTrack**.
    
    * **Algorytm:** Wykorzystuje Filtr Kalmana do przewidywania, gdzie obiekt będzie w kolejnej klatce.
    * **Dopasowanie:** Używa algorytmu węgierskiego (Hungarian Algorithm) do łączenia przewidywań z nowymi detekcjami.
    * **Zaleta:** Potrafi śledzić obiekty nawet przy niskim `confidence score` (np. gdy obraz jest rozmyty).

---

## Rozwiązywanie Problemów

!!! failure "Wyzwanie: Okluzja"
    W piłce nożnej zawodnicy często na siebie wbiegają. Wtedy detektor widzi "zlepioną" plamę, a tracker może zgubić ID (tzw. ID Switch).

!!! success "Nasze Rozwiązanie"
    Dzięki **ByteTrack**, system utrzymuje "martwe" ścieżki w pamięci przez 30 klatek. Jeśli zawodnik wyłoni się z tłumu w przewidywanym miejscu, odzyskuje swoje stare ID.