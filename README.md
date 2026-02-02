# Ilościowe Przewidywanie Zwrotów z Akcji

Projekt ten bada różne strategie uczenia maszynowego i arbitrażu statystycznego w celu przewidywania znaku dziennych zwrotów z akcji. Celem jest przekroczenie bazowej dokładności na poziomie 51,5% na trudnym zestawie danych finansowych o niskim sygnale.


| Plik Skryptu | Metoda | Dokładność | Kluczowa Mocna Strona |
|:---|:---|:---:|:---|
| **`advanced_clustering_ensemble.py`** | **DART + ExtraTrees + Clustering** | **51.38%** | Najlepsza generalizacja dzięki Dropout i Dynamicznemu Klastrowaniu. |
| **`shapelet_pattern_strategy.py`** | **Shapelets + Ranks** | **51.38%** | Wychwytuje specyficzne wzorce wykresów (V-kształtne, Trendy). |
| **`baseline_lightgbm.py`** | **LightGBM + Pair Features** | **51.35%** | Silny punkt odniesienia (baseline) używający prostych cech wartości wzglnej. |
| **`pca_ou_strategy.py`** | **PCA + Ornstein-Uhlenbeck** | **51.22%** | Czysta strategia powrotu do średniej oparta na resztach rynkowych. |
| **`rank_ensemble_model.py`** | **Ranked Ensemble** | **51.20%** | Wysoce stabilny dzięki rankingowi przekrojowemu (cross-sectional ranking). |

---

### 1. `pca_ou_strategy.py`: Arbitraż Statystyczny (Powrót do Średniej)

**Koncepcja:** 
Strategia ta zakłada, że ceny akcji składają się z komponentu "systematycznego" (napędzanego przez rynek/sektor) oraz komponentu "idiosynkratycznego" (unikalnego dla danej akcji). Komponent idiosynkratyczny jest modelowany jako proces Ornsteina-Uhlenbecka (OU) powracający do średniej. Kiedy akcja znacząco odchyla się od swojego punktu równowagi, obstawiamy jej powrót.

**Inżynieria Cech (Feature Engineering):**

1.  **Dekompozycja PCA:**
    Traktujemy macierz zwrotów $R$ ($T 	imes N$) jako mieszankę $K$ ukrytych czynników.
    $$r_{i,t} = \sum_{j=1}^{K} \beta_{i,j} F_{j,t} + \epsilon_{i,t}$$
    *   **Implementacja:** `sklearn.decomposition.PCA(n_components=20)` na przestawionej macierzy zwrotów.
    *   **Reszty ($\\epsilon_{i,t}$):** `Rzeczywiste Zwroty` - `Odwrotna Transformata(Czynniki PCA)`.

2.  **Skumulowane Reszty (Odchylenie Ceny):**
    Kumulujemy dzienne reszty, aby uzyskać serię odchyleń "poziomu ceny" $X_t$.
    $$X_{i,t} = \sum_{\\tau=0}^{t} \epsilon_{i,\tau}$$

3.  **Parametry Procesu Ornsteina-Uhlenbecka (OU):**
    Dopasowujemy dyskretną wersję SDE OU: $dX_t = \theta (\\mu - X_t)dt + \sigma dW_t$.
    Jest to rozwiązywane poprzez regresję liniową $X_t$ względem $X_{t-1}$ w oknie kroczącym ($W=60$):
    $$X_t = a + b X_{t-1} + \eta_t$$
    *   **Prędkość Powrotu do Średniej ($\\theta$):** $\\theta = 1 - b$
    *   **Poziom Równowagi ($\\mu$):** $\\mu = \frac{a}{1 - b}$
    *   **Sygnał (Z-Score):**
        $$Z_{score} = \frac{X_t - \mu}{\\text{std}(X_{t-W:t})}$$

**Model i Parametry:**
*   **Algorytm:** LightGBM (`LGBMClassifier`)
*   **Kluczowe Parametry:** `n_estimators=600`, `learning_rate=0.02`, `max_depth=6`, `num_leaves=31`.
*   **Cechy Wejściowe:** `RET_1..5`, `VOLUME_1..3`, `Rel_Market_Ret`, `Rel_Sector_Ret`, `PCA_Resid`, `OU_Theta`, `OU_Signal`.

---

### 2. `shapelet_pattern_strategy.py`: Dopasowywanie Wzorców Szeregów Czasowych

**Koncepcja:**
Tradycyjne modele ML często ignorują *kształt* ostatnich ruchów cenowych. To podejście wprost szuka klasycznych wzorców analizy technicznej (np. "Odbicie V-kształtne", "Głowa z Ramionami", "Trendy Momentum") poprzez porównywanie ostatnich okien cenowych do wyidealizowanych szablonów ("Shapelets").

**Inżynieria Cech (Feature Engineering):**

1.  **Z-Normalizacja:**
    Aby uczynić wzorce niezależnymi od skali (tak, aby akcja za 10$ i za 1000$ mogły pasować do tego samego kształtu), normalizujemy ostatnią sekwencję zwrotów $x$ w oknie $L$:
    $$\\hat{x} = \frac{x - \mu_x}{\\sigma_x}$$

2.  **Korelacja Shapelet (Podobieństwo):**
    Definiujemy idealne kształty $S$ (np. `V_Shape = [1, 0, -1, 0, 1]`). Cechą jest podobieństwo cosinusowe (iloczyn skalarny znormalizowanych wektorów):
    $$Feat_{corr} = \frac{1}{L} \sum_{i=1}^{L} \hat{x}_i \cdot \hat{S}_i$$
    
    **Zdefiniowane Kształty:**
    *   **Krótkoterminowe (5 dni):** `UpTrend` (Trend Wzrostowy), `DownTrend` (Trend Spadkowy), `V_Shape` (V-Kształt), `A_Shape` (A-Kształt), `Reversal_Up` (Odwrócenie w Górę), `Reversal_Down` (Odwrócenie w Dół).
    *   **Długoterminowe (20 dni):** `Long_Up` (Długi Wzrost), `Long_Down` (Długi Spadek), `U_Turn` (Zawracanie U).

3.  **Odległość Euklidesowa:**
    $$Feat_{dist} = ||\hat{x} - \hat{S}||_2$$

**Model i Parametry:**
*   **Ensemble:** VotingClassifier (Głosowanie Miękkie / Soft Voting)
    1.  **LightGBM:** `n_estimators=600`, `learning_rate=0.02`, `num_leaves=40`.
    2.  **ExtraTrees:** `n_estimators=200`, `max_depth=10` (wysoka losowość, aby zapobiec przeuczeniu).
*   **Wagi:** 2 (LGBM) : 1 (ExtraTrees).

---

### 3. `rank_ensemble_model.py`: Ranking Przekrojowy (Stacjonarność)

**Koncepcja:**
Dane finansowe są niezwykle zaszumione i niestacjonarne (zmienność zmienia się w czasie). Przekształcając wartości bezwzględne (zwroty, wolumen) na **rangi** (percentyle) względem wszystkich innych akcji w danym dniu, tworzymy solidny, stacjonarny zestaw danych. "Nie ma znaczenia, czy rynek się załamał; ważne jest, czy akcja A spadła mniej niż akcja B."

**Inżynieria Cech (Feature Engineering):**

1.  **Ranking Przekrojowy (Cross-Sectional Ranking):**
    Dla każdego dnia $t$, rangujemy wszystkie akcje $i \in \{1..N\}$ na podstawie cechy $f$:
    $$Rank_{i,t} = \frac{\text{Pozycja}(f_{i,t}) - 1}{N - 1} \in [0, 1]$$

2.  **Neutralność Sektorowa:**
    Usuwamy wpływ sektora, aby znaleźć najlepsze akcje *wewnątrz* sektora.
    $$RelRank_{i,t} = Rank_{i,t} - \frac{1}{|S_k|} \sum_{j \in S_k} Rank_{j,t}$$

3.  **Zmienność i Momentum Rankingu:**
    *   `R_Vol_5d`: Odchylenie standardowe rang dziennych zwrotów z 5 dni.
    *   `R_Mom_5d`: Średnia rang dziennych zwrotów z 5 dni.

**Model i Parametry:**
*   **Ensemble:** VotingClassifier
    1.  **LightGBM:** `n_estimators=400`, `learning_rate=0.03`.
    2.  **Random Forest:** `n_estimators=100`, `max_depth=8`.
*   **Dlaczego Random Forest?** Działa wyjątkowo dobrze z danymi dyskretnymi o wysokiej liczności i cechami rangowanymi, oferując różnorodność względem podejścia Gradient Boosting.

---

### 4. `advanced_clustering_ensemble.py`: Dynamiczne Klastrowanie i DART

**Koncepcja:**
Sektory (np. "Technologia") są szerokie i statyczne. Akcje często poruszają się razem w oparciu o ukryte czynniki, których nie wychwytują kody GICS. Ta strategia wykorzystuje Uczenie Nienadzorowane (K-Means) do odkrywania dynamicznych klastrów akcji, które obecnie zachowują się podobnie, a następnie mierzy siłę względem *tych* klastrów.

**Inżynieria Cech (Feature Engineering):**

1.  **Dynamiczne Klastrowanie (K-Means):**
    Klastrujemy akcje każdego dnia (lub w oknie kroczącym) na podstawie ich wektorów ostatnich zwrotów $\\mathbf{r}_i = [r_{t-1}, ..., r_{t-5}]$.
    $$C^* = \arg\min_{C} \sum_{k=1}^{K} \sum_{x \in C_k} ||x - \mu_k||^2$$
    *   **Implementacja:** `MiniBatchKMeans(n_clusters=50, batch_size=4096)`.

2.  **Siła Względna Klastra:**
    $$Alpha_{cluster} = Rank(Ret_{i,t}) - \text{Mean}(Rank(Ret_{j,t})) \quad \forall j \in Cluster(i)$$

**Model i Parametry:**
*   **Algorytm:** LightGBM z **DART** (Dropout Multiple Additive Regression Trees).
    *   **DART:** Losowo porzuca drzewa podczas treningu (podobnie jak Dropout w sieciach neuronowych). Zmusza to model, aby nie polegał na kilku dominujących cechach/drzewach, poprawiając generalizację.
    *   **Parametry:** `drop_rate=0.1`, `n_estimators=700`, `learning_rate=0.02`.
*   **Model Pomocniczy:** ExtraTreesClassifier (`n_estimators=300`) dla różnorodności strukturalnej.

---

### 5. `baseline_lightgbm.py`: Punkt Odniesienia Wartości Względnej

**Koncepcja:**
Solidny, prosty punkt odniesienia (baseline), który ustala minimalny poziom przewidywalności. Skupia się na koncepcjach "Pair Trading" — porównywaniu zwrotu z akcji do średniej rynkowej.

**Inżynieria Cech (Feature Engineering):**

1.  **Spread Rynkowy:**
    $$Spread_t = Ret_{i,t} - \text{ŚredniaRynku}_t$$
2.  **Spread Sektorowy:**
    $$Spread_{sector,t} = Ret_{i,t} - \text{ŚredniaSektora}_t$$

**Model:**
*   **LightGBM:** Standardowa implementacja GBDT.
*   **Wyniki:** Zaskakująco trudny do pobicia, ponieważ opiera się na najbardziej fundamentalnym prawie arbitrażu: powrocie do średniej względem indeksu.
