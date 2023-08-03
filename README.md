# Praca magisterska - Przewidywanie remodelingu oskrzeli w oparciu o dane mikromacierzowe
**Uczelnia: Uniwersytet Rzeszowski**  
**Kolegium: Kolegium Nauk Przyrodniczych**  
**Autor: Daniel Czyż**  
**Kierunek: Informatyka**  
**Praca wykonana pod kierunkiem: Dr hab. Jan Bazan, prof. UR**  

## Cel i teza pracy
**Celem pracy jest odpowiedzenie na jedno ważne pytanie, które brzmi następująco:**  
Czy przewidywanie remodelingu oskrzeli w oparciu o dane mikromacierzowe jest możliwe?  

Aby odpowiedzieć na postawione w pracy pytanie, należało udowodnić słuszność tezy.  

**Teza, jaką postawiono w pracy to:**  
Eksploracja danych mikromacierzowych DNA pozwoli na utworzenie modelu lub wielu modeli dobrej jakości, wykazując, że istnieje informacja płynąca z ekspresji genów zawarta w mikromacierzy DNA, która w dobrym stopniu wyjaśnia cechy kliniczne, wskazujące na występowanie remodelingu oskrzeli.  

## Uzasadnienie wyboru tematyki
**Innowacyjność:**  
W momencie doboru tematu pracy nie udało się znaleźć już istniejących publikacji badających dane mikromacierzowe DNA pod kątem przewidywania przypadłości remodelingu oskrzeli.  

Przeglądając dostępne publikacje można zauważyć, że mikromacierze DNA były i są nadal szeroko eksplorowane pod kątem przewidywania innych, bardziej popularnych przypadłości chorobowych, np. nowotwory, choroby układu krwionośnego.   

**Pozyskanie wiedzy przydatnej z punktu widzenia medycyny:**  
Gdyby wykazano możliwość przewidywania przypadłości remodelingu oskrzeli na podstawie danych mikromacierzowych, w przyszłości, takie modele dokonujące predykcji z wysoką skutecznością mogłyby być tworzone i stosowane jako część standardowej procedury diagnostycznej czy konsultacji z lekarzem.   

Informacja o zwiększonym ryzyku wystąpienia konkretnej choroby pozwoliłaby na lepsze przygotowanie się pacjenta, podjęcie czynności prewencyjnych lub zastosowanie odpowiedniego leczenia w odpowiednim czasie.  

## Zastosowana metodyka
1. **Wstępne przygotowanie danych**  
Usunięcie obiektów nieprzyjmujących wartości dla analizowanej kolumny decyzyjnej, progowanie jeśli przyjmowane wartości w kolumnie decyzyjnej były ciągłe.
2. **Walidacja krzyżowa (ang. Cross Validation)**  
Zastosowano mechanizm zagnieżdżonej walidacji krzyżowej typo LOO (Leave One Out). W wewnętrznej pętli walidacji krzyżowej dokonywano optymalizacji hiperparametrów metod, w zewnętrznej pętli odbywała się weryfikacja modelu na obiektach testowych.
3. **Wstępne przetwarzanie danych**  
W obrębie każdej pojedynczej iteracji walidacji krzyżowej przed przystąpieniem do nauczania modelu stosowano techniki wstępnego przetważania danych, które obejmowały kolejnio:
    1. **Skalowanie danych**  
    Sprowadzenie wartości w obrębie wszystkich obiektów dla każdej z kolumn do jedej skali. Zastosowano metodę MinMax. W wstępnych testach wykazano iż zastosowanie skalowania danych poprawia wyniki klasyfikacji.
    2. **Selekcja cech**  
    Zastosowano trzy metody selekcji cech. SelectKBest, SelectFromModel, RFE. Metodę SelectKBest stosowano w dwóch konfiguracjach, w pierwszej wykorzystując funkcję oceny cech chi2, w drugiej funkcję f_classif. Jako estymator przy metodzie SelectFromModel wykorzystano metodę RandomForestClassifier. Metodę RFE stosowano w dwóch konfiguracjach, w obu konfiguracjach wykorzystując jako estymator metodę LogisticRegression, lecz w pierwszej ustawiono metodę regularyzacji na metodę LASSO, a w drugiej konfiguracji na metodę RIDGE. 
    3. **Równoważenie klas decyzyjnych**  
    Gdy analizowana kolumna decyzyjna wymagała równoważenia klas, stosowano metodę SMOTE.
4. **Metody klasyfikujące**  
W pracy przetestowano dziesięć metod klasyfikacyjnych, tj. DecisionTree, RandomForest, GradientBoosting, MLP, LogisticRegression, GaussianNB, KNeighbours, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, SVC. 
5. **Weryfikacja jakości powstałego modelu**  
W procesie ewaluacji jak dany model radzi sobie z zadanym problemem przewidywania w badaniu wykorzystano takie metryki jak dokładność, precyzja, czułość, wskaźnik AUC, f-score weighted, zbalansowana dokładność.

## Eksperymenty
**Wykorzystane argumenty decyzyjne (cechy kliniczne):**
* Kolagen I % powierzchni,
* Kolagen I siła,
* Wall area ratio RB1,
* Wall area ratio RB10,
* Wall thichness/airway diameter ratio RB1,
* Średnia harmoniczna liniowa.

Wszystkie z wymienionych powyżej cech klinicznych są powiązane z występowaniem przypadłości remodelingu oskrzeli u osób chorych na astmę. Wykorzystanie właśnie tych cech klinicznych pacjentów jako argumenty decyzyjne było zdeterminowane przez lekarza specjalistę.  

### Wstępne uwagi:  
**Eksperymenty na próbkach ze szczotki** - w pracy skupiono się w części analizowania wyników na rezultatach eksploracji na próbkach z szczotki (swab samples). Wiedza o możliwości tworzenia skutecznych modeli przewidujących remodeling oskrzeli na próbkach z szczotki jest bardziej wartościową informacją z praktycznego punktu widzenia (łatwiej, bezpieczniej i taniej można pozyskać takie próbki w porównaniu z próbkami z krwi).  

**Hiperparametry metod oraz przestrzeń przeszukiwania** - ze względu na ograniczone zasoby czasu i mocy obliczeniowej skupiono się podczas eksploracji danych mikromacierzowych na optymalizacji hiperparametrów metod selekcji cech. Wszystkie hiperparametry metod klasyfikacyjnych na przestrzeni wszystkich eksperymentów pozostawiono przy ustawieniach domyślnych. Gdyby chciano badać hiperparamnetry metod klasyfikacyjnych czas wymagany na zakończenie badań bardzo szybko by się namnożył i uniemożliwiłby przeprowadzenie eksploracji w zaplanowanych ramach. Ponadto w wstępnych testach bardzo często to właśnie ustawienia domyślne okazywały się "zwycięskimi" konfiguracjami dla metod klasyfikacyjnych.  

**Poprawność skryptu/algorytmu** - przed przystąpieniem do eksploracji danych mikromacierzowych DNA zbadano czy utworzony skrypt działa poprawnie. W tym celu wykorzystano metodę "make_classification" z biblioteki scikit-learn. Metoda ta pozwala na wygenerowanie syntetycznego zbioru danych, którego trudność w kontekście problemu klasyfikacji można określić prametrami wejściowymi. Wyniki uzyskane za pomocą skryptu dla synetycznych danych pokrywały się z oczekiwaniami, więc następnie rozpoczęto badanie danych mikromacierzowych.  

## Wyniki dla Kolagen I % powierzchni (pięć najkorzystniejszych konfiguracji)
| Lp.  | Metoda klasyfikacyjna | Metoda selekcji cech    | Wal. Dokł.(%) | Test. Dokł.(%) | Test. Prec.(%) | Test. Czuł.(%) | Test. AUC.(%) |
|------|-----------------------|-------------------------|---------------|----------------|----------------|----------------|---------------|
| I.   | MLP                   | SelectKBest (f_classif) |     80.10     |      62.79     |      65.22     |      65.22     |     62.61     |
| II.  | GradientBoosting      | SelectKBest (f_classif) |     79.64     |      81.40     |      80.00     |      86.96     |     80.98     |
| III. | DecisionTree          | SelectKBest (f_classif) |     79.47     |      74.42     |      75.00     |      78.26     |     74.13     |
| IV.  | LinearDiscriminant    | SelectKBest (f_classif) |     79.25     |      69.77     |      72.72     |      69.57     |     69.78     |
| V.   | QuadraticDiscriminant |       RFE (LASSO)       |     78.28     |      74.42     |      73.08     |      82.60     |     73.80     |

## Wyniki dla Kolagen I siła (pięć najkorzystniejszych konfiguracji)
| Lp.  | Metoda klasyfikacyjna | Metoda selekcji cech | Wal. f_score weighted(%) | Test. Dokł.(%) | Test. Prec.(%) | Test. Czuł.(%) | Test. Bal. Dokł.(%) |
|------|-----------------------|----------------------|--------------------------|----------------|----------------|----------------|---------------------|
| I.   |          MLP          |      RFE (LASSO)     |           75.50          |      63.64     |      64.03     |      63.64     |        51.01        |
| II.  |          SVC          |      RFE (LASSO)     |           74.74          |      61.36     |      55.29     |      61.36     |        44.89        |
| III. |  Logistic Regression  |      RFE (LASSO)     |           73.66          |      59.09     |      57.95     |      59.09     |        47.41        |
| IV.  |     Random Forest     |      RFE (LASSO)     |           73.60          |      63.64     |      59.33     |      63.64     |        48.11        |
| V.   |       GaussianNB      |      RFE (LASSO)     |           73.28          |      70.45     |      62.55     |      70.45     |        53.79        |

## Wyniki dla Wall area ratio RB1 (pięć najkorzystniejszych konfiguracji)
| Lp.  | Metoda klasyfikacyjna  | Metoda selekcji cech    | Wal. Dokł.(%) | Test. Dokł.(%) | Test. Prec.(%) | Test. Czuł.(%) | Test. AUC.(%) |
|------|------------------------|-------------------------|---------------|----------------|----------------|----------------|---------------|
| I.   | Quadratic Discriminant |       RFE (RIDGE)       |     64.40     |      44.44     |      45.00     |      50.00     |     44.44     |
| II.  |      Decision Tree     |       RFE (RIDGE)       |     64.24     |      41.67     |      40.00     |      33.33     |     41.67     |
| III. | Quadratic Discriminant | SelectkBest (f_classif) |     63.42     |      38.89     |      40.00     |      44.44     |     38.89     |
| IV.  | Quadratic Discriminant |    SelectkBest (chi2)   |     62.53     |      55.56     |      55.00     |      61.11     |     55.56     |
| V.   |    Gradient Boosting   |       RFE (RIDGE)       |     61.55     |      27.78     |      27.78     |      27.78     |     27.78     |

## Wyniki dla Wall area ratio RB10 (pięć najkorzystniejszych konfiguracji)
| Lp.  | Metoda klasyfikacyjna | Metoda selekcji cech    | Wal. Dokł.(%) | Test. Dokł.(%) | Test. Prec.(%) | Test. Czuł.(%) | Test. AUC.(%) |
|------|-----------------------|-------------------------|---------------|----------------|----------------|----------------|---------------|
| I.   |       KNeighbors      | SelectKBest (f_classif) |     87.96     |      83.78     |      84.21     |      84.21     |     83.78     |
| II.  |       KNeighbors      |       RFE (LASSO)       |     87.27     |      86.49     |      85.00     |      89.47     |     86.40     |
| III. |   Gradient Boosting   | SelectKBest (f_classif) |     86.26     |      75.68     |      75.00     |      78.95     |     75.58     |
| IV.  |     Decision Tree     | SelectKBest (f_classif) |     85.95     |      78.38     |      78.95     |      78.95     |     78.36     |
| V.   |   Gradient Boosting   |       RFE (LASSO)       |     84.80     |      83.78     |      80.95     |      89.47     |     83.63     |

## Wyniki dla Wall thichness/airway diameter ratio RB10 (pięć najkorzystniejszych konfiguracji)
| Lp.  | Metoda klasyfikacyjna  | Metoda selekcji cech    | Wal. Dokł.(%) | Test. Dokł.(%) | Test. Prec.(%) | Test. Czuł.(%) | Test. AUC.(%) |
|------|------------------------|-------------------------|---------------|----------------|----------------|----------------|---------------|
| I.   | Quadratic Discriminant |       RFE (RIDGE)       |     64.40     |      44.44     |      45.00     |      50.00     |     44.44     |
| II.  |      Decision Tree     |       RFE (RIDGE)       |     64.24     |      41.67     |      40.00     |      33.33     |     41.67     |
| III. | Quadratic Discriminant | SelectkBest (f_classif) |     63.42     |      38.89     |      40.00     |      44.44     |     38.89     |
| IV.  | Quadratic Discriminant |    SelectkBest (chi2)   |     62.53     |      55.56     |      55.00     |      61.11     |     55.56     |
| V.   |    Gradient Boosting   |       RFE (RIDGE)       |     61.55     |      27.78     |      27.78     |      27.78     |     27.78     |

## Wyniki dla Średnia harmoniczna liniowa (pięć najkorzystniejszych konfiguracji)
| Lp.  | Metoda klasyfikacyjna  | Metoda selekcji cech    | Wal. Dokł.(%) | Test. Dokł.(%) | Test. Prec.(%) | Test. Czuł.(%) | Test. AUC.(%) |
|------|------------------------|-------------------------|---------------|----------------|----------------|----------------|---------------|
| I.   |    Gradient Boosting   |    SFM (RandomForest)   |     78.40     |      62.96     |      64.28     |      64.28     |     62.29     |
| II.  |      Decision Tree     |    SFM (RandomForest)   |     78.10     |      70.37     |      71.43     |      71.43     |     70.33     |
| III. |      Decision Tree     | SelectKBest (f_classif) |     74.11     |      62.96     |      62.50     |      71.43     |     62.64     |
| IV.  |   Linear Discriminant  |    SelectKBest (chi2)   |     72.33     |      59.26     |      58.82     |      71.43     |     58.79     |
| V.   | Quadratic Discriminant |    SelectKBest (chi2)   |     72.04     |      51.85     |      53.33     |      57.14     |     51.65     |

## Wnioski końcowe
Progiem akceptacji modelu jako dobrej jakości w badaniu był próg >80%. A więc każda metryka jakości dla danego modelu powinna mieć wartość powyżej lub równą 80%.  

Po przeanalizowaniu wszystkich z przeprowadzonych ekspertymentów tylko jedna kolumna decyzyjna pozwoliła uzyskać model dobrej jakości, tj. "Wall area ratio RB10".  

### Czy teza pracy została udowodniona?
Tak, lecz w kontekście tylko jednego argumentu decyzyjnego, tj. “Wall area ratio RB10”.  

Udowodniono, że eksploracja danych mikromacierzowych DNA pozwoli na utworzenie modelu lub wielu modeli dobrej jakości, wykazując, że istnieje informacja płynąca z ekspresji genów zawarta w mikromacierzy DNA, która w dobrym stopniu wyjaśnia cechę kliniczną “Wall area ratio RB10”, wskazującą na występowanie remodelingu oskrzeli.  

### A więc, czy przewidywanie remodelingu oskrzeli w oparciu o dane mikromacierzowe jest możliwe? 
Przeprowadzona eksploracja danych mikromacierzowych DNA sugeruje, że jest to możliwe w kontekście cechy klinicznej “Wall area ratio RB10”. W celu wykazania takiej możliwości w kontekście innych argumentów decyzyjnych należałoby dokonać dalszej eksploracji.  
