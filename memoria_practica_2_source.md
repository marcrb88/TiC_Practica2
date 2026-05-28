---
title: "Memòria de neteja i anàlisi del dataset de convocatòries públiques TIC del portal CIDO"
subtitle: "Pràctica 2 de Tipologia i cicle de vida de les dades"
author: "Marc Roige Benaiges · Eloi Vilella Escolano"
date: "Maig de 2026"
output:
  pdf_document:
    latex_engine: xelatex
    number_sections: true
    toc: false
geometry: margin=1in
fontsize: 10.5pt
mainfont: "Helvetica Neue"
header-includes:
  - \usepackage{booktabs}
  - \usepackage{array}
  - \usepackage{longtable}
  - \usepackage{xcolor}
  - \usepackage{float}
  - \definecolor{accent}{HTML}{2F4858}
  - \definecolor{softblue}{HTML}{EAF3F8}
  - \definecolor{softred}{HTML}{F8E9EC}
  - \renewcommand{\figurename}{Figura}
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(
  echo = FALSE,
  warning = FALSE,
  message = FALSE,
  fig.align = "center",
  fig.width = 8.6,
  fig.height = 4.8,
  out.width = "92%",
  fig.pos = "H",
  dev = "png",
  dpi = 160,
  fig.path = "memoria_practica_2_figures/"
)

library(tidyverse)
library(knitr)

df <- read_csv("cido_oposicions_analitzades.csv", show_col_types = FALSE)
validacio <- read_csv("validacio_xgboost_recerca.csv", show_col_types = FALSE)
shap_dir <- read_csv("direccio_shap_xgboost.csv", show_col_types = FALSE)
shap_valor <- read_csv("shap_per_valor_xgboost.csv", show_col_types = FALSE)
clusters <- read_csv("taula_clusters.csv", show_col_types = FALSE)
vars_cluster <- read_csv("variables_cluster_shap.csv", show_col_types = FALSE)

n_total <- nrow(df)
n_cols <- ncol(df)
n_recerca <- sum(df$perfil_recerca == "Recerca")
n_no_recerca <- sum(df$perfil_recerca == "No_recerca")
pct_recerca <- round(n_recerca / n_total * 100, 1)
pct_no_recerca <- round(n_no_recerca / n_total * 100, 1)

metriques <- validacio %>%
  summarise(
    `AUC mitjana` = mean(auc_test, na.rm = TRUE),
    `AUC mediana` = median(auc_test, na.rm = TRUE),
    `AUC Q1` = quantile(auc_test, 0.25, na.rm = TRUE),
    `AUC Q3` = quantile(auc_test, 0.75, na.rm = TRUE),
    `Accuracy mitjana` = mean(accuracy, na.rm = TRUE),
    `Sensibilitat mitjana` = mean(sensibilitat_recerca, na.rm = TRUE),
    `Especificitat mitjana` = mean(especificitat_no_recerca, na.rm = TRUE)
  ) %>%
  mutate(across(everything(), ~ round(.x, 3)))

auc_mitjana <- metriques[["AUC mitjana"]]
auc_mediana <- metriques[["AUC mediana"]]
accuracy_mitjana <- metriques[["Accuracy mitjana"]]
sensibilitat_mitjana <- metriques[["Sensibilitat mitjana"]]
especificitat_mitjana <- metriques[["Especificitat mitjana"]]

label_variable <- function(x) {
  recode(
    x,
    "tipus_personal_simplificatlaboral_temporal" = "Personal laboral temporal",
    "n_materies_cap" = "Nombre de matèries",
    "requereix_anglesTRUE" = "Requisit d'anglès",
    "requisit_catala_simplificatcatala_c1" = "Català C1",
    "requisit_catala_simplificatno_especificat" = "Català no especificat",
    "materia_informaticaTRUE" = "Matèria informàtica",
    "materia_enginyeriaTRUE" = "Matèria enginyeria",
    "grup_titulacio_simplificatuniversitari_a1" = "Titulació A1",
    .default = x
  )
}

label_categoria <- function(x) {
  recode(
    x,
    "universitari_a1" = "Universitari A1",
    "universitari_altres" = "Universitari altres",
    "no_universitari" = "No universitari",
    "merits" = "Mèrits",
    "prova_o_oposicio" = "Prova/oposició",
    "laboral_temporal" = "Laboral temporal",
    "laboral_no_temporal" = "Laboral no temporal",
    "funcionari_o_interi" = "Funcionari/interí",
    .default = x
  )
}

fmt_pct <- function(x) paste0(round(x, 1), "%")
```

# Dades bàsiques {.unnumbered}

- **Assignatura:** Tipologia i cicle de vida de les dades
- **Integrants:** Marc Roige Benaiges i Eloi Vilella Escolano
- **Font de dades:** portal CIDO d'oposicions i convocatòries públiques
- **Dataset original:** `cido_oposicions.csv`
- **Dataset final:** `cido_oposicions_analitzades.csv`
- **Repositori:** <https://github.com/marcrb88/TiC_Practica2.git>
- **Vídeo explicatiu:** pendent d'afegir l'enllaç de Google Drive abans del lliurament

# Resum executiu {.unnumbered}

Aquesta pràctica parteix del dataset generat a la Pràctica 1 i el transforma en un conjunt de dades preparat per fer anàlisi. L'objectiu no és només netejar el fitxer, sinó respondre una pregunta concreta: **quines característiques administratives, formatives i de requisits diferencien les ofertes vinculades a recerca de la resta d'ofertes TIC o informàtiques del CIDO?**

El dataset final conté `r n_total` ofertes i `r n_cols` variables. La variable objectiu, `perfil_recerca`, queda força equilibrada: `r n_recerca` registres de recerca (`r pct_recerca`%) i `r n_no_recerca` de no recerca (`r pct_no_recerca`%). Aquesta distribució és important perquè evita que el model aprengui només la classe majoritària.

La pràctica aplica un model supervisat XGBoost, interpreta el resultat amb SHAP values, construeix un agrupament k-means amb les variables més informatives i contrasta estadísticament la relació entre el perfil de recerca i el tipus de personal. El resultat és consistent: les ofertes de recerca presenten un patró diferenciat, especialment en el tipus de personal, el nombre de matèries associades i alguns requisits lingüístics.

# Descripció del dataset

El dataset recull convocatòries públiques publicades al CIDO relacionades amb perfils TIC, informàtica, tecnologia, enginyeria i recerca. Cada fila representa una oferta i conserva camps com el títol, l'estat, el sistema de selecció, el tipus de personal, el grup de titulació, les matèries i els requisits addicionals.

La pregunta de treball és:

\begin{quote}
Quines característiques administratives, formatives i de requisits diferencien les ofertes vinculades a recerca de la resta d'ofertes tecnològiques o informàtiques del CIDO?
\end{quote}

Aquesta pregunta és adequada perquè el dataset conté variables categòriques rellevants i perquè la classe de recerca no queda infrarepresentada. A més, permet combinar una lectura predictiva amb una interpretació substantiva del resultat.

```{r taula-resum-dataset}
tibble(
  Indicador = c(
    "Registres analitzats",
    "Variables finals",
    "Ofertes de recerca",
    "Ofertes de no recerca",
    "Repositori amb codi",
    "Fitxer final exportat"
  ),
  Valor = c(
    n_total,
    n_cols,
    paste0(n_recerca, " (", pct_recerca, "%)"),
    paste0(n_no_recerca, " (", pct_no_recerca, "%)"),
    "github.com/marcrb88/TiC_Practica2",
    "cido_oposicions_analitzades.csv"
  )
) %>%
  kable(booktabs = TRUE)
```

# Integració i selecció de dades

No s'ha integrat una font externa perquè el dataset original ja conté prou informació per respondre la pregunta: matèries, requisits, sistema de selecció, tipus de personal i titulació. La decisió principal ha estat seleccionar i transformar les variables útils, no afegir volum sense necessitat.

S'han descartat com a predictors els identificadors, les URL i les variables que podrien introduir informació massa directa o poc generalitzable. El target `perfil_recerca` s'ha derivat de les matèries, però `materia_recerca` no s'ha utilitzat com a predictor del model. Això evita una circularitat evident: no tindria sentit predir recerca fent servir directament l'indicador que defineix recerca.

```{r flow-diagram, fig.cap="Flux general de la pràctica", fig.width=10, fig.height=3.3}
nodes <- tribble(
  ~id, ~label, ~x, ~y,
  "raw", "Dataset original\nCIDO", 0, 1,
  "clean", "Neteja\nNA, dates i text", 2.2, 1,
  "features", "Variables derivades\nagrupació de classes", 4.4, 1,
  "xgb", "XGBoost\nRecerca / No_recerca", 6.6, 1.8,
  "shap", "SHAP\nimportància i signe", 8.8, 1.8,
  "kmeans", "k-means\nvariables SHAP", 6.6, 0.2,
  "concl", "Conclusions\nresposta al problema", 8.8, 0.2
)

edges <- tribble(
  ~x, ~y, ~xend, ~yend,
  0.75, 1, 1.45, 1,
  2.95, 1, 3.65, 1,
  5.15, 1.15, 5.85, 1.55,
  7.35, 1.8, 8.05, 1.8,
  5.15, 0.85, 5.85, 0.45,
  7.35, 0.2, 8.05, 0.2,
  8.8, 1.45, 8.8, 0.55
)

ggplot() +
  geom_segment(
    data = edges,
    aes(x = x, y = y, xend = xend, yend = yend),
    arrow = arrow(length = unit(0.18, "cm")),
    linewidth = 0.5,
    color = "#2F4858"
  ) +
  geom_label(
    data = nodes,
    aes(x = x, y = y, label = label),
    label.size = 0.25,
    fill = "#EAF3F8",
    color = "#1F2D35",
    size = 4.0,
    lineheight = 0.95,
    label.r = unit(0.1, "lines")
  ) +
  xlim(-0.7, 9.6) +
  ylim(-0.35, 2.35) +
  theme_void()
```

# Neteja de les dades

La neteja s'ha organitzat en quatre blocs: tractament de valors buits, conversió de tipus, gestió de valors extrems i creació de variables derivades.

Els valors buits i el literal `Vegeu les bases` s'han tractat com a valors absents, perquè no aporten informació analítica. Les dates s'han convertit a format temporal extraient la primera data vàlida del camp. També s'ha normalitzat el text a minúscules i sense accents per detectar patrons de manera estable.

```{r variables-creades}
tibble(
  Bloc = c(
    "Requisits",
    "Matèries",
    "Titulació",
    "Tipus de personal",
    "Sistema de selecció",
    "Temporalitat i volum"
  ),
  Variables = c(
    "`requereix_angles`, `requereix_catala`, `requisit_catala_simplificat`",
    "`materia_informatica`, `materia_tecnologia`, `materia_enginyeria`, `perfil_recerca`",
    "`grup_titulacio_simplificat`",
    "`tipus_personal_simplificat`",
    "`sistema_simplificat`",
    "`n_materies`, `n_materies_cap`, `mes_final`, `te_data_final`"
  ),
  Justificacio = c(
    "Permeten analitzar requisits lingüístics i d'idiomes.",
    "Converteixen text temàtic en indicadors modelables.",
    "Agrupa A, A1 i A2 en categories més estables.",
    "Redueix categories petites i separa temporalitat laboral.",
    "Diferencia mèrits de processos amb prova o oposició.",
    "Resumeix informació temporal i evita pes excessiu de valors alts."
  )
) %>%
  kable(booktabs = TRUE)
```

El nombre de matèries per oferta s'ha revisat com a possible valor extrem. La variable original té valors entre `r min(df$n_materies)` i `r max(df$n_materies)`, amb mediana `r median(df$n_materies)`. Per evitar que les ofertes amb més etiquetes temàtiques tinguin massa pes, s'ha creat `n_materies_cap`, limitada amb la regla de l'IQR.

```{r distribucio-target, fig.cap="Distribució de la variable objectiu"}
df %>%
  count(perfil_recerca) %>%
  mutate(percentatge = n / sum(n)) %>%
  ggplot(aes(x = perfil_recerca, y = n, fill = perfil_recerca)) +
  geom_col(width = 0.62) +
  geom_text(aes(label = paste0(n, " (", scales::percent(percentatge, accuracy = 0.1), ")")), vjust = -0.35, size = 3.5) +
  scale_fill_manual(values = c("No_recerca" = "#86BBD8", "Recerca" = "#2F4858")) +
  labs(x = NULL, y = "Nombre d'ofertes") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none")
```

# Anàlisi de les dades

## Model supervisat: XGBoost

El model supervisat utilitzat és XGBoost, amb classificació binària de `perfil_recerca`. Els predictors inclouen titulació simplificada, tipus de personal, sistema de selecció, requisits lingüístics, matèries no circulars i variables temporals.

Per evitar que el resultat depengui d'una única partició train/test, s'han executat 100 particions estratificades. En cada iteració s'ha entrenat el model amb validació creuada interna i s'ha avaluat sobre el conjunt de test.

```{r metriques-model}
metriques %>%
  pivot_longer(everything(), names_to = "Metrica", values_to = "Valor") %>%
  kable(booktabs = TRUE, digits = 3)
```

```{r auc-plot, fig.cap="Distribució de l'AUC en 100 particions train/test"}
validacio %>%
  ggplot(aes(x = auc_test)) +
  geom_histogram(binwidth = 0.05, fill = "#86BBD8", color = "white", boundary = 0.5) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "#D1495B") +
  geom_vline(xintercept = auc_mediana, color = "#2F4858", linewidth = 0.9) +
  labs(
    x = "AUC en test",
    y = "Nombre de particions"
  ) +
  theme_minimal(base_size = 11)
```

El model obté una AUC mitjana de `r auc_mitjana` i una AUC mediana de `r auc_mediana`. Això indica que les variables seleccionades tenen capacitat real per separar ofertes de recerca i no recerca. L'accuracy mitjana és `r accuracy_mitjana`, amb sensibilitat `r sensibilitat_mitjana` i especificitat `r especificitat_mitjana`.

\newpage

## Interpretació amb SHAP

L'ús de SHAP és clau perquè el model no quedi només com una predicció numèrica. La importància absoluta mostra quines variables pesen més; el signe del SHAP indica si la variable empeny cap a `Recerca` o cap a `No_recerca`.

```{r shap-taula}
shap_dir %>%
  select(variable, mean_abs_shap, mean_shap, direccio_mitjana) %>%
  slice_head(n = 8) %>%
  mutate(variable = label_variable(variable)) %>%
  mutate(
    mean_abs_shap = round(mean_abs_shap, 3),
    mean_shap = round(mean_shap, 3)
  ) %>%
  kable(
    booktabs = TRUE,
    col.names = c("Variable", "SHAP absolut mitjà", "SHAP mitjà signat", "Direcció mitjana")
  )
```

```{r shap-signe, fig.cap="Direcció mitjana dels SHAP values"}
shap_dir %>%
  slice_head(n = 8) %>%
  mutate(variable = fct_reorder(label_variable(variable), mean_shap)) %>%
  ggplot(aes(x = variable, y = mean_shap, fill = mean_shap > 0)) +
  geom_col(width = 0.72) +
  coord_flip() +
  geom_hline(yintercept = 0, color = "grey45") +
  scale_fill_manual(values = c("#D1495B", "#2F4858"), guide = "none") +
  labs(x = NULL, y = "SHAP mitjà signat") +
  theme_minimal(base_size = 10)
```

Les variables més importants són el personal laboral temporal, el nombre de matèries, el requisit d'anglès i els indicadors del requisit de català. La lectura principal és que el perfil de recerca no es diferencia només per la matèria, sinó també per la forma administrativa de l'oferta.

\newpage

## Model no supervisat: k-means guiat per SHAP

El k-means s'ha aplicat només sobre les variables més rellevants segons SHAP. Aquesta decisió fa que el model no supervisat sigui més interpretable: no agrupa les ofertes amb totes les variables possibles, sinó amb les dimensions que el model supervisat ha detectat com a més informatives.

```{r clusters-taula}
clusters %>%
  transmute(
    Cluster = cluster,
    `N` = n,
    `% recerca` = pct_recerca,
    `% anglès` = pct_angles,
    `% català C1` = pct_catala_c1,
    `Mitjana matèries` = mitjana_materies,
    `Titulació majoritària` = label_categoria(grup_titulacio_majoritari),
    `Sistema majoritari` = label_categoria(sistema_majoritari),
    `Tipus majoritari` = label_categoria(tipus_majoritari)
  ) %>%
  kable(booktabs = TRUE, digits = 2)
```

El resultat mostra tres perfils. El cluster 2 és clarament menys vinculat a recerca, amb només `r clusters$pct_recerca[clusters$cluster == 2]`% d'ofertes de recerca. El cluster 3 concentra més recerca, amb `r clusters$pct_recerca[clusters$cluster == 3]`%. Això dona una lectura complementària al model supervisat: quan es miren les variables més rellevants, apareixen grups amb proporcions de recerca diferents.

## Contrast d'hipòtesi

La prova estadística s'ha plantejat sobre una relació concreta:

- **H0:** el perfil de recerca i el tipus de personal simplificat són independents.
- **H1:** el perfil de recerca i el tipus de personal simplificat estan associats.

Com que la taula de contingència pot tenir freqüències petites, s'ha utilitzat el test exacte de Fisher quan no es compleixen les condicions del khi quadrat. El p-valor obtingut és inferior a 0,001, de manera que es rebutja la hipòtesi nul·la al 5%.

La conclusió és que el tipus de personal no es distribueix igual entre ofertes de recerca i no recerca. Aquest resultat encaixa amb la interpretació SHAP, on el tipus de personal apareix com una de les variables amb més pes.

# Representació dels resultats

Les representacions s'han distribuït al llarg de la memòria per evitar concentrar totes les figures al final. S'han utilitzat taules de resum, gràfics de barres, histogrames de validació, SHAP signat i un diagrama de flux del procés analític.

```{r perfils-personal, fig.cap="Perfil de recerca segons tipus de personal"}
df %>%
  count(perfil_recerca, tipus_personal_simplificat) %>%
  group_by(tipus_personal_simplificat) %>%
  mutate(percentatge = n / sum(n)) %>%
  mutate(tipus_personal_simplificat = label_categoria(tipus_personal_simplificat)) %>%
  ggplot(aes(x = tipus_personal_simplificat, y = percentatge, fill = perfil_recerca)) +
  geom_col(position = "fill", width = 0.72) +
  coord_flip() +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(values = c("No_recerca" = "#86BBD8", "Recerca" = "#2F4858")) +
  labs(x = NULL, y = "Percentatge", fill = "Perfil") +
  theme_minimal(base_size = 10)
```

# Resolució del problema

Els resultats permeten respondre la pregunta inicial. Les ofertes vinculades a recerca es poden distingir de la resta a partir de característiques administratives, formatives i de requisits.

La primera evidència és predictiva: XGBoost obté una AUC mitjana de `r auc_mitjana` en 100 particions, molt per sobre del valor 0,5 que representaria una classificació sense capacitat discriminant. La segona evidència és interpretativa: SHAP mostra que el tipus de personal, el nombre de matèries i alguns requisits lingüístics expliquen bona part de la classificació. La tercera evidència és descriptiva: el k-means identifica grups amb proporcions de recerca molt diferents. Finalment, el contrast d'hipòtesi confirma una associació estadísticament significativa entre el perfil de recerca i el tipus de personal.

La conclusió s'ha d'entendre com una lectura exploratòria, perquè el dataset és petit i prové d'una instantània del portal. Tot i això, els diferents mètodes apunten al mateix resultat: les ofertes de recerca no són simplement una etiqueta temàtica dins del CIDO, sinó que tenen un patró administratiu propi.

# Codi i fitxers lliurats

El codi principal de la pràctica és `TiC_Pract2.Rmd`. Aquest fitxer conté la neteja, el feature engineering, l'entrenament del model XGBoost, la interpretació SHAP, el k-means, el contrast d'hipòtesi, les visualitzacions i l'exportació dels fitxers finals.

```{r fitxers}
tibble(
  Fitxer = c(
    "TiC_Pract2.Rmd",
    "cido_oposicions.csv",
    "cido_oposicions_analitzades.csv",
    "taula_clusters.csv",
    "importancia_shap_xgboost.csv",
    "direccio_shap_xgboost.csv",
    "shap_per_valor_xgboost.csv",
    "validacio_xgboost_recerca.csv",
    "memoria_practica_2.pdf"
  ),
  Funcio = c(
    "Codi principal de la pràctica",
    "Dataset original",
    "Dataset final net i enriquit",
    "Resum dels clusters",
    "Importància global de variables",
    "Direcció mitjana dels SHAP values",
    "SHAP mitjà segons valor de variable",
    "Resultats de les 100 particions del model",
    "Memòria final en PDF"
  )
) %>%
  kable(booktabs = TRUE)
```

# Vídeo

El vídeo explicatiu ha de resumir la pràctica en un màxim de 10 minuts i ha d'incloure la participació dels dos integrants. Es recomana seguir aquest guió:

1. Presentació del dataset i de la pregunta de recerca.
2. Neteja i variables derivades.
3. Resultats del model XGBoost.
4. Interpretació SHAP.
5. Clustering i contrast d'hipòtesi.
6. Conclusions i limitacions.

**Enllaç al vídeo:** pendent d'afegir abans del lliurament.

# Taula de contribucions

```{r contribucions}
tibble(
  Contribucio = c(
    "Investigació prèvia",
    "Redacció de les respostes",
    "Desenvolupament del codi",
    "Participació al vídeo"
  ),
  Signatura = c(
    "MRB, EVE",
    "MRB, EVE",
    "MRB, EVE",
    "MRB, EVE"
  )
) %>%
  kable(booktabs = TRUE)
```

# Bibliografia i recursos

- Calvo, M., Subirats, L. i Pérez, D. (2019). *Introducción a la limpieza y análisis de los datos*. Editorial UOC.
- Han, J., Kamber, M. i Pei, J. (2012). *Data mining: concepts and techniques*. Morgan Kaufmann.
- Osborne, J. W. (2010). *Data Cleaning Basics: Best Practices in Dealing with Extreme Scores*.
- Chen, T. i Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*.
- Lundberg, S. M. i Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*.
