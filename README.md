# Cats and Dogs Project
Copyright (C) 2019 Oleksii Kosobutskyi

Program używa sieć neuronową do rozpoznawania szczekania i miauczenia.

Program wymaga zewnętrznych bibliotek wymienionych w requirements.txt.
Instalacja bibliotek: pip install nazwa_biblioteki

Struktura sieci MLP:
- warstwa całkowicie połączona, 256 neuronów, funkcja aktywacji relu
- dropout (losowe usunięcie neuronów)
- warstwa całkowicie połączona, 256 neuronów, funkcja aktywacji relu
- dropout (losowe usunięcie neuronów)
- warstwa całkowicie połączona, 2 neurony, funkcja aktywacji softmax

Przykład użycia: python projekt.py nazwa_pliku_testowego
