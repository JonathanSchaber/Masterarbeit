# Assessment of SRL quality

A "-"-dashed line symbolizes sentence boundaries inside sequences.
A "="-dashed line indicates the boundary between sequences in so-called sentence pair tasks, or the boundary between context and question in Q&A tasks, respectively.

Note that the tagging scheme follows the **B**eginning, **I**nside, **O**utside notation; tokens in sentences for which no predicate was identified are annotated **0** (zero).

The tags for predicates (`B-V`) are surrounded by asterisks for faster identification.

For each example, please check in the file `assessment.md`, if you estimate the SRLs to be

- **helpful** (if, e.g. the SRL analysis conforms to the actual state of affairs and potentially helps the model to identify relevant aspects of the task --- e.g. it correctly identifies agents in  active/passive alternations)
- **neutral** (if, e.g. the SRL analysis is not very precise but also not wrong in the strict sense. The model could probably not get any useful information out of it)

or

- **harmful** (if, e.g. the SRL analysis is plain false and the SRLs would probably confuse the model by drawing its attention to the wrong parts of the sentences --- e.g. the identified predicate is wrong or active/passive alternations are not recognized)

for the task.

## deISEAR

Task: Predict the correct emotion for [MASK] (out of seven emotions)

### 1

```
Ich        B-A0         O
fühlte     *B-V*        O
[MASK]     B-A1         O
,          I-A1         O
als        I-A1         O
ich        I-A1         B-A0
aus        I-A1         O
Versehen   I-A1         O
schlechte  I-A1         B-A1
Milch      I-A1         I-A1
getrunken  I-A1         *B-V*
habe       I-A1         O
```


### 2

```
Ich      B-A0           O
fühlte   *B-V*          O
[MASK]   B-A1           O
,        I-A1           O
weil     I-A1           O
mir      I-A1           B-A0
mal      I-A1           O
der      I-A1           B-A1
Sprit    I-A1           I-A1
ausging  I-A1           *B-V*
.        O              O
```


### 3

```
Ich              B-A0           O               O
fühle            *B-V*          O               O
[MASK]           B-A1           O               O
,                I-A1           O               O
weil             I-A1           O               O
sich             I-A1           O               O
Deutsche         I-A1           B-A0            B-A0
im               I-A1           I-A0            I-A0
Ausland          I-A1           I-A0            I-A0
zu               I-A1           O               O
oft              I-A1           O               O
daneben          I-A1           O               O
benehmen         I-A1           *B-V*           O
und              I-A1           O               O
sich             I-A1           O               O
nicht            I-A1           O               O
den              I-A1           O               B-A1
lokalen          I-A1           O               I-A1
Gepflogenheiten  I-A1           O               I-A1
anpassen         I-A1           O               *B-V*
können           I-A1           O               O
.                O              O               O
```


### 4

```
Ich      B-A0           O
fühlte   *B-V*          O
[MASK]   B-A1           O
,        I-A1           O
als      I-A1           O
ich      I-A1           B-A0
alleine  I-A1           B-A1
in       I-A1           I-A1
einem    I-A1           I-A1
Wald     I-A1           I-A1
war      I-A1           O
.        O              O
```


### 5

```
Ich             B-A0            O               O
fühlte          *B-V*           O               O
[MASK]          B-A1            O               O
,               I-A1            O               O
als             I-A1            O               O
mir             I-A1            B-A2            O
gesagt          I-A1            *B-V*           O
wurde           I-A1            O               O
,               I-A1            O               O
dass            I-A1            B-A1            O
ich             I-A1            I-A1            B-A0
Weihnachtsgeld  I-A1            I-A1            B-A1
erhalten        I-A1            I-A1            *B-V*
würde           I-A1            I-A1            O
.               O               O               O
```

### 6

```
Ich      B-A0           O               O
fühlte   *B-V*          O               O
[MASK]   B-A1           O               O
,        I-A1           O               O
als      I-A1           O               O
ich      I-A1           B-A0            B-A0
mt       I-A1           *B-V*           B-A1
einer    I-A1           B-A1            I-A1
fremden  I-A1           I-A1            I-A1
Frau     I-A1           I-A1            I-A1
Sex      I-A1           I-A1            I-A1
hatte    I-A1           O               *B-V*
.        O              O               O
```

### 7

```
Ich       B-A0          O               O
fühlte    *B-V*         O               O
[MASK]    B-A1          O               O
,         I-A1          O               O
als       I-A1          O               O
ich       I-A1          B-A0            B-A0
meinen    I-A1          B-A2            B-A2
kleinen   I-A1          I-A1            I-A2
Sohn      I-A1          I-A2            I-A2
die       I-A1          B-A2            B-A1
ersten    I-A1          I-A2            I-A1
Worte     I-A1          I-A2            I-A1
sprechen  I-A1          *B-V*           I-A1
hörte     I-A1          O               *B-V*
.         O             O               O
```


### 8

```
Ich            B-A0             O               O
fühlte         *B-V*            O               O
[MASK]         B-A1             O               O
,              I-A1             O               O
weil           I-A1             O               O
jemand         I-A1             B-A0            B-A0
sein           I-A1             B-A1            B-A1
verschmutztes  I-A1             I-A1            I-A1
Taschentuch    I-A1             I-A1            I-A1
auf            I-A1             B-A2            B-A1
dem            I-A1             I-A2            I-A1
Arbeitstisch   I-A1             I-A2            I-A1
liegen         I-A1             *B-V*           I-A1
ließ           I-A1             O               *B-V*
.              O                O               O
```


### 9

```
Ich               B-A0          O
fühlte            *B-V*         O
[MASK]            B-A1          O
,                 I-A1          O
als               I-A1          O
ich               I-A1          B-A0
aus               I-A1          O
gesundheitlichen  I-A1          O
Gründen           I-A1          O
mein              I-A1          B-A1
Hobby             I-A1          I-A1
aufgeben          I-A1          *B-V*
musste            I-A1          O
.                 O             O
```


### 10

```
Ich         B-A0                O
fühlte      *B-V*               O
[MASK]      B-A1                O
,           I-A1                O
als         I-A1                O
ich         I-A1                B-A0
mit         I-A1                O
meinem      I-A1                O
Auto        I-A1                O
ohne        I-A1                O
Bedrängnis  I-A1                O
gegen       I-A1                B-A4
eine        I-A1                I-A4
Mauer       I-A1                I-A4
gefahren    I-A1                *B-V*
bin         I-A1                O
.           O                   O
```


### 11

```
Ich         B-A0                O
fühlte      *B-V*               O
[MASK]      B-A1                O
,           I-A1                O
als         I-A1                O
mein        I-A1                B-A0
bester      I-A1                I-A0
Freund      I-A1                I-A0
meinen      I-A1                B-A1
Geburtstag  I-A1                I-A1
vergeseen   I-A1                I-A1
hatte       I-A1                *B-V*
.           O                   O
```


### 12

```
Ich             B-A0            O
fühlte          *B-V*           O
[MASK]          B-A1            O
,               I-A1            O
als             I-A1            O
ich             I-A1            B-A0
verschimmeltes  I-A1            O
Essen           I-A1            O
im              I-A1            O
Kühlschrank     I-A1            O
gefunden        I-A1            *B-V*
habe            I-A1            O
.               O               O
```


### 13

```
Ich      B-A0
fühlte   *B-V*
[MASK]   B-A1
,        I-A1
als      I-A1
mein     I-A1
2        I-A1
.        O
------------------------------
Kind     B-A0
geboren  *B-V*
wurde    O
.        O
```



### 14

```
Ich          B-A0               O               O               O
fühlte       *B-V*              O               O               O
[MASK]       B-A1               O               O               O
,            I-A1               O               O               O
als          I-A1               O               O               O
ich          I-A1               B-A0            B-A0            O
es           I-A1               B-A1            O               O
nicht        I-A1               O               O               O
mehr         I-A1               O               O               O
geschafft    I-A1               *B-V*           O               O
habe         I-A1               O               O               O
zum          I-A1               B-C-A1          B-A3            O
Friseur      I-A1               I-C-A1          I-A3            O
zu           I-A1               I-C-A1          O               O
gehen        I-A1               I-C-A1          *B-V*           O
uns          I-A1               I-C-A1          O               B-A0
meine        I-A1               I-C-A1          O               B-A1
Frisur       I-A1               I-C-A1          O               I-A1
schrecklich  I-A1               I-C-A1          O               I-A1
aussah       I-A1               I-C-A1          O               *B-V*
```


### 15

```
Ich          B-A0               O
fühlte       *B-V*              O
[MASK]       B-A1               O
,            I-A1               O
als          I-A1               O
ich          I-A1               B-A0
einen        I-A1               B-A1
Hundehaufen  I-A1               I-A1
gelaufen     I-A1               *B-V*
bin          I-A1               O
.            O                  O
```


### 16

```
Ich        B-A0         O
fühlte     *B-V*        O
[MASK]     B-A1         O
,          I-A1         O
als        I-A1         B-A1
ich        I-A1         B-A0
einen      I-A1         I-A1
Bericht    I-A1         I-A1
über       I-A1         I-A1
Spinnen    I-A1         I-A1
im         I-A1         I-A1
Fernsehen  I-A1         I-A1
gesehen    I-A1         *B-V*
habe       I-A1         O
.          O            O
```


### 17

```
Ich                 B-A0                O
fühlte              *B-V*               O
[MASK]              B-A1                O
,                   I-A1                O
als                 I-A1                O
ich                 I-A1                B-A0
im                  I-A1                O
Radio               I-A1                O
einen               I-A1                B-A1
Bericht             I-A1                I-A1
über                I-A1                I-A1
DarkNet             I-A1                I-A1
gehört              I-A1                *B-V*
habe                I-A1                O
.                   O                   O
```


### 18

```
Ich         B-A0        O               O
fühlte      *B-V*       O               O
[MASK]      B-A1        O               O
,           O           O               O
nachdem     O           O               O
meine       O           B-A0            O
Mutter      O           I-A0            O
mir         O           B-A2            O
sagte       O           *B-V*           O
,           O           O               O
dass        O           B-A1            O
mein        O           I-A1            B-A0
Vater       O           I-A1            I-A0
verstorben  O           I-A1            *B-V*
ist         O           I-A1            O
.           O           O               O
```


### 19

```
Ich            B-A0             O               O
fühlte         *B-V*            O               O
[MASK]         B-A1             O               O
,              I-A1             O               O
als            I-A1             O               O
ich            I-A1             B-A0            O
meinen         I-A1             B-A1            O
zweijährigen   I-A1             I-A1            O
Neffen         I-A1             O               O
versehentlich  I-A1             O               O
gegen          I-A1             B-A2            O
einen          I-A1             I-A2            O
Stuhl          I-A1             I-A2            O
geschubst      I-A1             *B-V*           O
habe           I-A1             O               O
und            I-A1             O               O
er             I-A1             O               B-A0
sich           I-A1             O               O
den            I-A1             O               B-A1
Kopf           I-A1             O               I-A1
angeschlagen   I-A1             O               *B-V*
hatte          I-A1             O               O
.              O                O               O
```


### 20

```
Ich            B-A0             O               O               O               O
fühlte         *B-V*            O               O               O               O
[MASK]         B-A1             O               O               O               O
,              I-A1             O               O               O               O
als            I-A1             O               O               O               O
ich            I-A1             B-A0            B-A0            O               O
nachts         I-A1             O               O               O               O
auf            I-A1             O               O               O               O
dem            I-A1             O               O               O               O
Weg            I-A1             O               O               O               O
nach           I-A1             B-A1            O               O               O
Hause          I-A1             I-A1            O               O               O
war            I-A1             *B-V*           O               O               O
und            I-A1             O               O               O               O
merkte         I-A1             O               *B-V*           O               O
,              I-A1             O               O               O               O
dass           I-A1             O               B-A1            O               O
eine           I-A1             O               I-A1            B-A1            O
Gruppe         I-A1             O               I-A1            I-A1            O
junger         I-A1             O               I-A1            I-A1            O
Männer         I-A1             O               I-A1            I-A1            O
im             I-A1             O               I-A1            O               O
angetrunkenen  I-A1             O               I-A1            O               O
Zustand        I-A1             O               I-A1            O               O
in             I-A1             O               I-A1            B-A0            O
der            I-A1             O               I-A1            I-A0            O
Unterführung   I-A1             O               I-A1            I-A0            O
standen        I-A1             O               I-A1            *B-V*           O
,              I-A1             O               O               O               O
die            I-A1             O               O               B-C-A0          B-A1
ich            I-A1             O               O               O               B-A0
durchqueren    I-A1             O               O               O               *B-V*
musste         I-A1             O               O               O               O
.              O                O               O               O               O
```



## SCARE

Task: Predict the correct sentiment for a review (positive, negative, neutral)

Note that the two vertical lines, «||» denote the boundary between review title and content.

### 1

```
Doff       B-A1
|          I-A1
|          I-A1
Stürzt     *B-V*
dauerhaft  O
ab         O
```


### 2

```
Bester       0
Player       0
|            0
|            0
Meiner       0
Meinung      0
nach         0
der          0
beste        0
Player       0
im           0
Market       0
.            0
```


### 3

```
Was          B-A1
ist          *B-V*
jetzt        O
los          B-A0
?            O
```


### 4

```
Gut        O
auch       O
im         O
Urlaub     O
und        O
bei        O
Radtouren  O
|          O
|          O
Mit        O
dieser     O
APP        O
kann       O
man        B-A0
Radtouren  B-A3
besser     O
planen     *B-V*
.          O
```


### 5

```
Naja           O                O               O
|              O                O               O
|              O                O               O
Die            B-A0             O               O
App            I-A0             O               O
startet        *B-V*            O               O
sich           B-A1             O               O
laufend        O                O               O
selbst         O                O               O
im             O                O               O
Hintergrund    O                O               O
,              O                O               O
nichtmals      O                O               O
mit            O                O               O
'              O                O               O
nem            O                O               O
Task           O                O               O
-              O                O               O
killer         O                B-A1            O
unter          O                B-A2            O
Kontrolle      O                I-A2            O
zu             O                O               O
bringen        O                *B-V*           O
,              O                O               O
Akku           O                O               O
schnell        O                O               O
leer           O                O               O
.              O                O               O
Nach           O                O               O
nicht          O                O               O
mal            O                O               O
einem          O                O               O
Tag            O                O               O
wieder         O                O               O
deinstalliert  O                O               O
.              O                O               O
Schade         O                O               O
,              O                O               O
denn           O                O               O
die            O                O               B-A0
Oberfläche     O                O               B-A1
und            O                O               I-A1
Funktionen     O                O               I-A0
sind           O                O               *B-V*
großartig      O                O               O
.              O                O               O
```


### 6

```
Einfach       O
geil          O
|             O
|             O
Die           B-A1
beste         I-A1
Entscheidung  I-A1
dieses        I-A1
Jahr          I-A1
Spotify       O
zu            O
nutzen        *B-V*
diese         O
App           O
seit          O
langer        O
zeit          O
.             O
```


### 7

```
Nach        O
Update      O
,           O
nimmt       *B-V*
es          B-A0
nicht       O
alle        B-A1
Buchstaben  I-A1
[UNK]       I-A1
```


### 8

```
Ok  0
|   0
|   0
Ok  0
```


### 9

```
Beste     B-A1
ist       *B-V*
facebook  O
|         O
|         O
Nö        O
```


### 10

```
perfekt     B-A1
|           I-A1
|           I-A1
Leute       I-A1
vielleicht  O
liegt       *B-V*
es          O
an          B-A2
euren       I-A2
Handys      I-A2
[UNK]       O
```


### 11

```
st          *B-V*              O
zuverlässig  O          O
so           O          O
gut          O          O
es           O          O
geht         O          *B-V*
```


### 12

```
Ganz       O            O               O
okay       O            O               O
|          O            O               O
|          O            O               O
Also       O            O               O
ich        B-A0         B-A0            O
brauch     *B-V*        O               O
die        B-A1         B-A2            O
app        I-A1         I-A2            O
nur        O            O               O
um         O            O               O
mit        O            O               O
einer      O            O               O
person     O            O               O
zu         O            O               O
schreiben  O            *B-V*           O
und        O            O               O
dafür      O            O               O
langt      O            O               *B-V*
sie        O            O               B-A1
auch       O            O               O
.          O            O               O
```


### 13

```
Samsung  0
galaxy   0
s5       0
|        0
|        0
Joar     0
```


### 14

```
Fantastische  B-A0
App           O
,             O
aber          O
nach          O
wie           O
vor           O
zuviele       O
abstürze      *B-V*
bei           O
einzelnen     O
Reporten      O
.             O
```


### 15

```
Handy    B-A1
top      I-A1
Tablet   I-A1
flop     I-A1
|        I-A1
|        I-A1
Seit     I-A1
dem      I-A1
Update   I-A1
komme    *B-V*
ich      B-A0
nur      O
noch     O
auf      B-A0
die      I-A0
1        I-A0
.        O
Seite    O
.        O
```


### 16

```
Twitter      0
?            0
```


### 17

```
Toll  B-A1
|     I-A1
|     I-A1
Ist   *B-V*
cool  O
```


### 18

```
Einfach  0
gut      0
.        0
.        0
.        0
.        0
```


### 19

```
Nicht         O         O               O
schlecht      O         O               O
,             O         O               O
gut           O         O               O
aufgebaut     *B-V*     O               O
|             O         O               O
|             O         O               O
Leider        O         O               O
kann          O         O               O
man           O         B-A0            O
die           O         B-A1            O
Eilmeldungen  O         I-A1            O
nicht         O         O               O
auf           O         B-A1            O
seine         O         I-A1            O
Bedürfnisse   O         I-A1            O
einstellen    O         *B-V*           O
,             O         O               O
den           O         B-C-A1          B-A2
nicht         O         I-C-A1          O
alles         O         I-C-A1          B-A0
ist           O         I-C-A1          *B-V*
von           O         I-C-A1          B-A1
Interesse     O         I-C-A1          I-A1
.             O         O               O
```


### 20

```
Unfertiges   0
Spiel        0
|            0
|            0
Viele        0
Bugs         0
,            0
Sound        0
bugt         0
rum          0
,            0
schrecklich  0
Steuerung    0
```



## XNLI

Task: Predict the entailment status for a premise, hypothesis pair (entailment, contradiction, neutral)

### 1

```
Die             B-A1            O               O               O
Aufgabe         I-A1            O               O               O
welchen         I-A1            O               O               O
Aspekts         I-A1            O               O               O
unserer         I-A1            O               O               O
Außenpolitik    I-A1            O               O               O
fürchtet        *B-V*           O               O               O
Richard         B-A0            O               O               O
Clarke          I-A0            O               O               O
am              B-C-A1          O               O               O
meisten         I-C-A1          O               O               O
-               O               O               O               O
-               O               O               O               O
herumstehen     O               O               O               O
,               O               O               O               O
während         O               O               O               O
Zivilisten      O               B-A1            B-A1            O
in              O               O               O               O
Ruanda          O               O               O               O
abgeschlachtet  O               *B-V*           O               O
werden          O               O               O               O
oder            O               O               O               O
herumstehen     O               O               *B-V*           O
,               O               O               O               O
während         O               O               O               O
Zivilisten      O               O               O               B-A1
im              O               O               O               O
Kosovo          O               O               O               O
abgeschlachtet  O               O               O               *B-V*
werden          O               O               O               O
?               O               O               O               O
==============================
Clarke     O            O
sorgt      *B-V*        O
sich       O            O
darum      B-A0         O
,          I-A0         O
wie        I-A0         O
wir        I-A0         B-A0
auf        I-A0         B-A1
die        I-A0         I-A1
neuste     I-A0         I-A1
Gewalt     I-A0         I-A1
antworten  I-A0         *B-V*
werden     I-A0         O
.          O            O
```


### 2

```
Ich      B-A0
hab      *B-V*
noch     O
platz    B-A1
für      I-A1
sechs    I-A1
Schotch  I-A1
'        O
s        O
==============================
Ich       B-A0
habe      *B-V*
nur       B-A1
Platz     I-A1
für       I-A1
ein       I-A1
weiteres  I-A1
Glas      I-A1
Scotch    I-A1
.         O
```


### 3

```
Es                B-A0          O
gibt              *B-V*         O
zwei              B-A1          O
evolutionäre      I-A1          O
Vorteile          I-A1          O
für               I-A1          O
Menschen          I-A1          O
die               I-A1          O
durchschnittlich  I-A1          O
aussehen          I-A1          *B-V*
.                 O             O
==============================
Es                O
ist               O
in                O
Ordnung           O
durchschnittlich  B-A1
auszusehen        *B-V*
.                 O
```


### 4

```
Es             B-A0             O
gibt           *B-V*            O
keinen         B-A1             O
Grund          I-A1             O
sich           I-A1             O
für            I-A1             O
unser          I-A1             O
Verhalten      I-A1             O
als            I-A1             O
Leiter         I-A1             O
zu             I-A1             O
entschuldigen  I-A1             *B-V*
.              O                O
==============================
Unsere        B-A0              O
einzige       I-A0              O
Möglichkeit   I-A0              O
besteht       *B-V*             O
darin         B-A1              O
,             I-A1              O
den           I-A1              B-A1
Anweisungen   I-A1              I-A1
unserer       I-A1              I-A1
Vorgesetzten  I-A1              I-A1
zu            I-A1              O
folgen        I-A1              *B-V*
.             O                 O
```


### 5

```
In             O
Tokio          O
entdeckte      *B-V*
ein            B-A0
Korrespondent  I-A0
der            I-A0
Zeitung        I-A0
The            I-A0
Economist      I-A0
ein            B-A1
T              I-A1
-              I-A1
Shirt          I-A1
mit            I-A1
der            I-A1
Aufschrift     I-A1
O              I-A1
D              I-A1
on             I-A1
Bourgeoisie    I-A1
Milk           I-A1
Boy            I-A1
Milk           I-A1
.              O
==============================
Ein         B-A0                B-A0
Reporter    I-A0                I-A0
von         I-A0                I-A0
The         I-A0                I-A0
Economist   I-A0                I-A0
war         *B-V*               O
in          O                   O
Tokio       O                   O
und         O                   O
bemerkte    O                   *B-V*
ein         O                   B-A1
bestimmtes  O                   I-A1
T           O                   I-A1
-           O                   I-A1
Shirt       O                   I-A1
.           O                   O
```


### 6

```
Es           O          O               O
war          O          O               O
das          O          O               O
Wichtigste   O          O               O
was          B-A1       O               O
wir          B-A0       O               O
sichern      *B-V*      O               O
wollten      O          O               O
da           O          O               O
es           O          O               O
keine        O          B-A1            O
Möglichkeit  O          I-A1            O
gab          O          *B-V*           O
eine         O          B-A1            B-A3
20           O          I-A1            I-A3
Megatonnen   O          I-A1            I-A3
-            O          I-A1            I-A3
H            O          I-A1            I-A3
-            O          I-A1            I-A3
Bombe        O          I-A1            I-A3
ab           O          I-A1            O
zu           O          I-A1            B-A5
werfen       O          I-A1            *B-V*
von          O          I-A1            I-A5
einem        O          I-A1            I-A5
30           O          I-A3            I-A5
,            O          O               O
C124         O          O               O
.            O          O               O
==============================
Wir         B-A0
wollten     O
eine        B-A1
Sache       I-A1
mehr        I-A1
retten      *B-V*
als         B-C-A1
die         I-C-A1
Restlichen  I-C-A1
.           O
```


### 7

```
Ich        B-A0         O
weiß       *B-V*        O
nicht      O            O
ob         B-A1         O
er         I-A1         B-A0
danach     I-A1         O
in         I-A1         B-A1
Augusta    I-A1         I-A1
geblieben  I-A1         *B-V*
ist        I-A1         O
.          O            O
==============================
Er         B-A0
wohnte     *B-V*
weiterhin  O
in         B-A1
Augusta    I-A1
.          O
```


### 8

```
Inglish        O
unterscheidet  *B-V*
sich           O
von            B-A0
Englisch       I-A0
durch          I-A0
fünf           I-A0
Wörter         I-A0
,              I-A0
Ausdrücke      I-A0
,              I-A0
Grammatik      I-A0
,              I-A0
Aussprache     I-A0
und            I-A0
Rythmus        I-A0
.              O
==============================
Inglish   O
ist       *B-V*
etwas     B-A0
anderes   I-A0
als       O
Englisch  O
.         O
```


### 9

```
Wenn        O           O               O               O
Esperanto   O           O               O               O
bestrebt    O           O               O               O
ist         O           O               O               O
eine        O           O               O               O
echte       O           O               O               O
Sprache     O           O               O               O
zu          O           O               O               O
werden      *B-V*       O               O               O
,           O           O               O               O
dann        O           O               O               O
muss        O           O               O               O
es          O           B-A0            O               O
anfangen    O           *B-V*           O               O
,           O           O               O               O
sich        O           O               O               O
wie         O           B-A1            O               O
eine        O           I-A1            B-A1            O
Sprache     O           I-A1            I-A1            O
zu          O           I-A1            O               O
verhalten   O           I-A1            *B-V*           O
,           O           O               O               O
und         O           O               O               O
wird        O           O               O               O
dann        O           O               O               O
bald        O           O               O               O
die         O           O               O               O
gleichen    O           O               O               O
Schwächen   O           O               O               O
erleiden    O           O               O               O
,           O           O               O               O
die         O           O               O               B-A0
natürliche  O           O               O               B-A1
Sprachen    O           O               O               I-A1
erleiden    O           O               O               *B-V*
-           O           O               O               O
Polysemie   O           O               O               O
und         O           O               O               O
Pleuritis   O           O               O               O
.           O           O               O               O
==============================
Polysemie   O           O
ist         *B-V*       O
ein         B-A1        O
Problem     I-A1        O
,           I-A1        O
das         I-A1        B-A1
natürliche  I-A1        B-A0
Sprachen    I-A1        I-A0
haben       I-A1        *B-V*
.           O           O
```


### 10

```
Weißt       *B-V*       O               O
du          B-A0        O               O
,           O           O               O
du          B-A1        O               O
kannst      I-A1        O               O
nicht       I-A1        O               O
,           I-A1        O               O
du          I-A1        B-A0            O
kannst      I-A1        B-A1            O
nicht       I-A1        O               O
überleben   I-A1        *B-V*           O
,           I-A1        O               O
wenn        I-A1        O               O
du          I-A1        O               B-A0
keinen      I-A1        O               O
Gegendruck  I-A1        O               O
hast        I-A1        O               O
,           I-A1        O               O
um          I-A1        O               O
den         I-A1        O               B-A3
Atemdruck   I-A1        O               I-A3
in          I-A1        O               I-A3
diesen      I-A1        O               I-A3
Höhen       I-A1        O               I-A3
zu          I-A1        O               O
erhöhen     I-A1        O               *B-V*
.           O           O               O
==============================
Sie         B-A0
benötigen   *B-V*
einen       B-A3
Gegendruck  I-A3
über        I-A3
5000        I-A3
Fuß         I-A3
.           O
```




## PAWS-X

Task: Predict if two sentences paraphrase each other (true, false)

### 1

```
Die            I-A2
61             I-A2
aufgelisteten  I-A2
Immobilien     I-A2
und            I-A2
historischen   I-A2
Viertel        I-A2
in             I-A2
Evanston       I-A2
und            I-A2
für            I-A2
die            I-A2
mehr           I-A2
als            I-A2
350            I-A2
gelisteten     I-A2
Immobilien     I-A2
und            I-A2
Viertel        I-A2
in             O
Chicago        O
werden         O
separate       O
Listen         O
erstellt       *B-V*
.              O
==============================
Für           O
die           O
61            O
gelisteten    O
Immobilien    O
-             O
und           O
historischen  O
Viertel       O
in            O
Chicago       O
sowie         O
für           O
mehr          O
als           O
350           O
gelistete     O
Immobilien    O
und           O
Viertel       O
in            O
Evanston      O
werden        O
separate      O
Listen        O
erstellt      *B-V*
.             O
```


### 2

```
Der             B-A0            B-A0
Charakter       I-A0            I-A0
von             I-A0            I-A0
Holden          I-A0            I-A0
Ford            I-A0            I-A0
basiert         *B-V*           I-A0
auf             O               I-A0
FBI             O               I-A0
-               O               I-A0
Agent           O               I-A0
John            O               I-A0
E               O               I-A0
.               O               O
Douglas         O               O
,               O               O
und             O               O
Bill            O               O
Tench           O               O
basiert         O               *B-V*
auf             O               O
dem             O               O
bahnbrechenden  O               O
FBI             O               O
-               O               O
Agenten         O               O
Robert          O               O
K               O               O
.               O               O
Ressler         O               O
.               O               O
==============================
Der             B-A0            B-A0
Charakter       I-A0            I-A0
von             I-A0            I-A0
Holden          I-A0            I-A0
Ford            I-A0            I-A0
basiert         *B-V*           I-A0
auf             O               I-A0
dem             O               I-A0
FBI             O               I-A0
-               O               I-A0
Agenten         O               I-A0
John            O               I-A0
E               O               I-A0
.               O               O
Douglas         O               O
,               O               O
und             O               O
Bill            O               O
Tench           O               O
basiert         O               *B-V*
auf             O               B-A1
dem             O               I-A1
bahnbrechenden  O               I-A1
FBI             O               I-A1
-               O               O
Agenten         O               O
Robert          O               O
K               O               O
.               O               O
Ressler         O               O
.               O               O
```


### 3

```
Andrew                    B-A1
W                         I-A1
.                         O
Mellon                    O
ist                       *B-V*
Literaturwissenschaftler  B-A0
,                         I-A0
Kritiker                  I-A0
und                       I-A0
Romanschriftsteller       I-A0
und                       I-A0
derzeit                   I-A0
Professor                 I-A0
für                       I-A0
Distinguished             I-A0
Service                   I-A0
an                        O
der                       O
University                O
of                        O
Chicago                   O
von                       O
Frederick                 O
A                         O
.                         O
------------------------------
de                        0
Armas                     0
.                         0
==============================
Frederick                 0
A                         0
.                         0
------------------------------
de                        B-A1
Armas                     I-A1
ist                       *B-V*
Literaturwissenschaftler  B-A0
,                         O
Kritiker                  O
und                       O
Romanschriftsteller       O
und                       O
zurzeit                   O
Andrew                    O
W                         O
.                         O
Mellon                    O
Distinguished             O
Service                   O
Professor                 O
für                       O
Geisteswissenschaften     O
an                        O
der                       O
University                O
of                        O
Chicago                   O
.                         O
```


### 4

```
2014             B-A1           O
brachten         *B-V*          O
iOS              O              O
und              O              O
Android          O              O
Applikationen    O              O
zur              O              O
Produktsuche     O              O
heraus           O              O
;                O              O
Produktfeatures  O              O
beinhalten       O              *B-V*
interaktive      O              B-A0
Video            O              I-A0
-                O              I-A0
Produktreviews   O              I-A0
mit              O              O
live             O              O
Frage            O              O
-                O              O
und              O              O
Antwort          O              O
-                O              O
Sessions         O              O
.                O              O
==============================
Im                       O
Jahr                     O
2014                     O
startete                 *B-V*
die                      B-A3
Website                  I-A3
iOS                      I-A3
und                      I-A3
Android                  I-A3
-                        I-A3
Anwendungen              I-A3
für                      I-A3
die                      I-A3
Produktsuche             I-A3
.                        O
```


### 5

```
Es              O               O
wurde           O               O
von             B-A0            B-A0
seiner          I-A0            I-A0
Frau            I-A0            I-A0
,               I-A0            I-A0
Stella          I-A0            I-A0
Gemmell         I-A0            I-A0
,               O               O
nach            O               O
seinem          O               O
Tod             O               O
am              O               O
28              O               O
.               O               O
Juli            O               O
2006            O               B-A2
fertiggestellt  *B-V*           I-A2
und             O               I-A2
unter           O               O
der             O               O
gemeinsamen     O               O
Autorschaft     O               O
von             O               O
David           O               O
und             O               O
Stella          O               O
Gemmell         O               O
veröffentlicht  O               *B-V*
.               O               O
==============================
Es              B-A2            B-A1
wurde           O               O
nach            O               O
seinem          O               O
Tod             O               O
am              O               O
28              O               O
.               O               O
Juli            O               O
2006            O               O
von             B-A0            O
seiner          I-A0            O
Ehefrau         I-A0            O
Stella          I-A0            O
Gemmell         I-A0            O
veröffentlicht  *B-V*           O
und             O               O
unter           O               B-A2
der             O               I-A2
gemeinsamen     O               I-A2
Autorenschaft   O               I-A2
von             O               I-A2
David           O               I-A2
und             O               I-A2
Stella          O               I-A2
Gemmell         O               I-A2
fertig          O               O
gestellt        O               *B-V*
.               O               O
```


### 6

```
Die         B-A0
erste       I-A0
Runde       I-A0
fand        *B-V*
am          B-A3
Wochenende  I-A3
vom         B-A1
23          I-A1
.           O
bis         O
25          O
.           O
September   O
2011        O
in          O
der         O
Slowakei    O
(           O
Prievidza   O
)           O
statt       O
.           O
==============================
Die         B-A0
erste       I-A0
Runde       I-A0
fand        *B-V*
am          B-A3
Wochenende  I-A3
vom         B-A1
23          I-A1
.           O
bis         O
25          O
.           O
September   O
2011        O
in          O
der         O
Slowakei    O
(           O
Prievidza   O
)           O
statt       O
.           O
```


### 7

```
Elizabeth    O          O
Prudden      O          O
immigrierte  *B-V*      O
1653         O          O
von          B-A0       O
England      I-A0       O
nach         O          O
Milford      O          O
,            O          O
Connecticut  O          O
;            O          O
sie          O          B-A0
heiratete    O          *B-V*
Roger        O          B-A1
Pritchard    O          I-A1
.            O          O
==============================
Elizabeth    O          B-A1
Elizabeth    O          I-A1
Prudden      O          I-A1
,            O          I-A1
1653         O          I-A1
aus          O          I-A1
England      O          I-A1
nach         B-A3       I-A1
Milford      I-A3       I-A1
emigriert    *B-V*      I-A1
,            O          I-A1
Connecticut  O          I-A1
,            O          O
heiratete    O          *B-V*
Roger        O          B-A0
Pritchard    O          I-A0
```


### 8

```
Während      O          O               O
der          O          O               O
Zeit         O          O               O
der          O          O               O
japanischen  O          O               O
Herrschaft   O          O               O
wurde        O          O               O
Namaxia      B-A1       B-A1            O
mit          B-A2       O               O
dem          I-A2       O               O
Maolin       I-A2       O               O
District     I-A2       O               O
und          I-A2       O               O
dem          I-A2       O               O
Taoyuan      I-A2       O               O
District     I-A2       O               O
gruppiert    *B-V*      O               O
und          O          O               O
als          O          O               O
regiert      O          *B-V*           O
,            O          O               O
der          O          O               B-A1
dem          O          O               I-A1
Kizan        O          O               I-A1
District     O          O               I-A1
der          O          O               I-A1
Takao        O          O               I-A1
-            O          O               O
Präfektur    O          O               O
zugeordnet   O          O               *B-V*
wurde        O          O               O
.            O          O               O
==============================
Während      O                  O
der          O                  O
Zeit         O                  O
der          O                  O
japanischen  O                  O
Herrschaft   O                  O
wurde        O                  O
Namaxia      B-A1               O
in           B-A3               O
den          I-A3               O
Bezirk       I-A3               O
Maolin       I-A3               O
und          I-A3               O
den          I-A3               O
Bezirk       I-A3               O
Taoyuan      O                  O
eingeteilt   *B-V*              O
,            O                  O
der          B-C-A1             B-A0
unter        I-C-A0             O
dem          I-C-A0             O
Bezirk       I-C-A0             O
Kizan        I-C-A0             O
in           I-C-A0             B-A1
der          I-C-A0             I-A1
Präfektur    I-C-A0             I-A1
Takao        I-C-A0             I-A1
stand        I-C-A0             *B-V*
.            O                  O
```


### 9

```
Arabic               O          O               O
Supplement           O          O               O
ist                  *B-V*      O               O
ein                  B-A1       O               O
Unicode              I-A1       O               O
-                    I-A1       O               O
Block                I-A1       O               O
,                    I-A1       O               O
der                  I-A1       B-A0            O
arabische            I-A1       B-A1            O
Buchstabenvarianten  I-A1       I-A1            O
kodiert              I-A1       *B-V*           O
,                    I-A1       O               O
die                  I-A1       O               B-A1
zum                  I-A1       O               O
Schreiben            I-A1       O               O
von                  I-A1       O               O
nichtarabischen      I-A1       O               O
Sprachen             I-A1       O               O
verwendet            I-A1       O               *B-V*
werden               I-A1       O               O
,                    O          O               O
einschließlich       O          O               O
Sprachen             O          O               O
von                  O          O               O
Pakistan             O          O               O
und                  O          O               O
Afrika               O          O               O
sowie                O          O               O
von                  O          O               O
altpersischen        O          O               O
.                    O          O               O
==============================
Bei                  O          O               O
der                  O          O               O
Ergänzung            O          O               O
handelt              *B-V*      O               O
es                   O          O               O
sich                 O          O               O
um                   B-A0       O               O
einen                I-A0       O               O
Unicode              I-A0       O               O
-                    I-A0       O               O
Block                I-A0       O               O
,                    I-A0       O               O
der                  I-A0       B-A1            O
arabische            I-A0       I-A1            O
Buchstabenvarianten  I-A0       I-A1            O
kodiert              I-A0       *B-V*           O
,                    I-A0       O               O
die                  I-A0       O               B-A3
zum                  I-A0       O               O
Schreiben            I-A0       O               O
von                  I-A0       O               O
nichtarabischen      I-A0       O               O
Sprachen             I-A0       O               O
verwendet            I-A0       O               *B-V*
werden               I-A0       O               O
,                    I-A0       O               O
einschließlich       I-A0       O               O
der                  I-A0       O               O
Sprachen             I-A0       O               O
Pakistans            I-A0       O               O
und                  I-A0       O               O
Afrikas              I-A0       O               O
sowie                I-A0       O               O
des                  I-A0       O               O
alten                I-A0       O               O
Persischen           I-A0       O               O
.                    O          O               O
```


### 10

```
Sie                     B-A1
war                     *B-V*
Mutter                  B-A0
von                     I-A0
Val                     I-A0
,                       I-A0
Boris                   I-A0
und                     I-A0
Rosalind                I-A0
Lorwin                  I-A0
.                       O
==============================
Sie                     O
wurde                   O
Mutter                  O
von                     O
Val                     O
,                       O
Boris                   O
und                     O
Rosalind                O
Lorwin                  O
,                       O
ihre                    B-A1
Tochter                 I-A1
war                     *B-V*
Psychologieprofessorin  I-A0
.                       O
```


## MLQA

Task: Predict answer span in a context given the context and a question

### 1

```
Während             O       O
seiner              O       O
Amtszeit            O       O
nahm                *B-V*   O
die                 B-A1    O
Christenverfolgung  I-A1    O
zu                  O       O
,                   O       O
obwohl              O       O
neue                O       B-A3
Kirchen             O       I-A3
und                 O       I-A3
Friedhöfe           O       I-A3
gebaut              O       *B-V*
werden              O       O
konnten             O       O
.                   O       O
------------------------------
Wahrscheinlich      O       O       O
starb               *B-V*   O       O
er                  B-A0    O       O
nicht               O       O       O
als                 B-A3    O       O
Märtyrer            I-A3    O       O
,                   O       O       O
denn                O       O       O
die                 O       B-A0    O
Christenverfolgung  O       I-A0    O
Diokletians         O       I-A0    O
begann              O       *B-V*   O
erst                O       O       O
im                  O       O       O
Jahre               O       O       O
303                 O       O       O
,                   O       O       O
also                O       O       O
nach                O       B-A3    O
Cajus               O       I-A3    O
[UNK]               O       I-A3    O
mutmaßlichem        O       I-A3    O
Tod                 O       I-A3    O
,                   O       O       O
und                 O       O       O
Diokletian          O       O       O
war                 O       O       O
auf                 O       O       O
seinem              O       O       O
Weg                 O       O       O
zum                 O       O       O
Kaiser              O       O       O
den                 O       O       O
Christen            O       O       O
gegenüber           O       O       O
nicht               O       O       O
immer               O       O       O
feindlich           O       O       O
eingestellt         O       O       *B-V*
.                   O       O       O
------------------------------
Als          O       O
Bischof      O       O
von          O       O
Rom          O       O
verfügte     *B-V*   O
Cajus        O       O
,            O       O
dass         B-A1    O
ein          I-A1    B-A0
zukünftiger  I-A1    I-A0
Bischof      I-A1    I-A0
zuerst       I-A1    I-A0
Lektor       I-A1    I-A0
,            I-A1    I-A0
Exorzist     I-A1    I-A0
,            I-A1    I-A0
Akolyth      I-A1    I-A0
,            I-A1    I-A0
Subdiakon    I-A1    I-A0
,            I-A1    I-A0
Diakon       I-A1    I-A0
und          I-A1    I-A0
Priester     I-A1    I-A0
gewesen      I-A1    *B-V*
sein         I-A1    B-A1
musste       I-A1    I-A1
.            O       O
==============================
Welche       B-A1       B-A2
Positionen   I-A1       I-A2
muss         O          O
man          B-A0       B-A0
erreichen    *B-V*      O
,            O          O
um           O          O
die          O          B-A1
von          O          I-A1
Kaius        O          I-A1
angeordnete  O          I-A1
Position     O          I-A1
eines        O          I-A1
Läufers      O          O
einzunehmen  O          *B-V*
?            O          O
```


### 2

```
Mit                O
dem                O
überraschenden     O
,                  O
die                O
Vermeidung         O
eines              O
Zweifrontenkriegs  O
begünstigenden     O
Hitler             O
-                  O
Stalin             O
-                  O
Pakt               O
erschien           *B-V*
Hitler             B-A0
der                B-A0
Überfall           O
auf                O
Polen              O
als                O
ein                B-A1
überschaubares     O
Risiko             O
.                  O
------------------------------
Am         O
1          O
.          O
September  O
1939       O
begann     *B-V*
das        B-A1
Deutsche   I-A1
Reich      I-A1
mit        O
dem        O
Überfall   O
auf        B-A0
Polen      O
den        O
Zweiten    O
Weltkrieg  O
.          O
------------------------------
Der            B-A1    O
Blitzkrieg     I-A1    O
war            *B-V*   O
von            B-A0    O
Polen          I-A0    O
über           O       O
Norwegen       O       O
und            O       O
im             O       O
Westfeldzug    O       O
so             O       O
erfolgreich    O       O
,              O       O
dass           O       O
Hitler         O       O
trotz          O       O
der            O       O
am             O       O
energischen    O       O
Widerstand     O       O
unter          O       O
Winston        O       O
Churchill      O       O
gescheiterten  O       O
Luftschlacht   O       O
um             O       O
England        O       O
am             O       O
22             O       O
.              O       O
Juni           O       O
1941           O       O
das            O       B-A0
Unternehmen    O       O
Barbarossa     O       O
und            O       O
den            O       B-A1
darauf         O       B-A4
folgenden      O       I-A1
Krieg          O       I-A1
gegen          O       I-A1
die            O       I-A1
Sowjetunion    O       I-A1
befahl         O       *B-V*
.              O       O
------------------------------
Der             B-A1
deutsche        I-A1
Vormarsch       I-A1
wurde           O
von             B-A0
der             I-A0
weit            I-A0
unterschätzten  I-A0
Roten           I-A0
Armee           I-A0
mit             O
Einbruch        O
des             O
Winters         O
in              O
der             O
Schlacht        O
um              O
Moskau          O
gestoppt        *B-V*
.               O
------------------------------
Doch            O
auch            O
den             O
gerade          O
nach            O
dem             O
japanischen     O
Angriff         O
auf             O
Pearl           O
Harbor          O
in              O
den             O
Krieg           O
eingetretenen   O
USA             O
erklärte        *B-V*
Hitler          B-A0
am              O
11              O
.               O
Dezember        O
1941            O
deutscherseits  O
den             B-A1
Krieg           I-A1
.               O
------------------------------
Die                      B-A1
auf                      I-A1
[UNK]                    I-A1
Lebensraum               I-A1
[UNK]                    I-A1
-                        I-A1
Eroberung                I-A1
gerichtete               I-A1
militärische             I-A1
Ostexpansion             I-A1
des                      I-A1
nationalsozialistischen  I-A1
Deutschland              B-A0
sah                      *B-V*
auch                     O
für                      O
die                      O
einheimische             O
Zivilbevölkerung         O
keinerlei                B-A2
Schonung                 I-A2
vor                      O
.                        O
------------------------------
Vielmehr          O       O
zielten           *B-V*   O
Zwangsarbeit      O       O
und               O       O
Aushungern        O       O
auf               B-A1    O
eine              I-A1    O
radikale          I-A1    O
Dezimierung       I-A1    O
der               I-A1    O
slawischen        I-A1    O
[UNK]             I-A1    O
Untermenschen     I-A1    O
[UNK]             I-A1    O
,                 I-A1    O
an                I-A1    O
deren             I-A1    O
Stelle            I-A1    O
arische           I-A1    B-A0
[UNK]             I-A1    I-A0
Herrenmenschen    I-A1    I-A0
[UNK]             I-A1    I-A0
als               I-A1    O
Kolonisten        I-A1    O
in                I-A1    B-A1
einem             I-A1    O
künftigen         I-A1    O
[UNK]             I-A1    O
Großgermanischen  I-A1    O
Reich             I-A1    O
[UNK]             I-A1    O
herrschen         I-A1    *B-V*
sollten           I-A1    O
.                 O       O
------------------------------
Im                O       O
Generalplan       O       O
Ost               O       O
war               O       O
die               O       B-A1
[UNK]             O       I-A1
Verschrottung     O       I-A1
[UNK]             O       I-A1
von               O       I-A1
31                O       I-A1
Millionen         O       I-A1
Slawen            O       I-A1
vorgesehen        O       *B-V*
,                 O       O
im                O       O
Protokoll         O       O
der               O       O
Wannseekonferenz  *B-V*   O
die               B-A1    O
Vernichtung       I-A1    O
von               I-A1    O
11                I-A1    O
Millionen         I-A1    O
Juden             I-A1    O
im                I-A1    O
Rahmen            I-A1    O
des               I-A1    O
Holocaust         I-A3    O
.                 O       O
------------------------------
Zwischen        O
1941            O
und             O
1944            O
stieg           *B-V*
die             B-A1
Zahl            I-A1
der             B-A1
nach            I-A1
Deutschland     I-A1
verschleppten   I-A1
Zwangsarbeiter  I-A1
von             B-A4
drei            I-A4
auf             B-A3
acht            I-A3
Millionen       I-A3
.               O
------------------------------
Das                  B-A0    O       O
dem                  B-A1    O       O
Vernichtungslager    I-A1    O       O
Auschwitz            I-A1    O       O
-                    O       O       O
Birkenau             O       O       O
angeschlossene       O       O       O
Zwangsarbeiterlager  O       O       O
Auschwitz            O       O       O
-                    O       O       O
Monowitz             O       O       O
gehörte              *B-V*   O       O
zum                  O       O       O
oberschlesischen     O       O       O
Chemie               O       O       O
-                    O       O       O
Komplex              O       O       O
,                    O       O       O
der                  O       B-A0    O
Dimensionen          O       B-A1    O
annahm               O       *B-V*   O
,                    O       O       O
die                  O       B-C-A1  B-A1
denen                O       I-C-A1  B-A2
des                  O       I-C-A1  I-A1
Ruhrgebiets          O       I-C-A1  I-A1
kaum                 O       I-C-A1  O
nachstanden          O       I-C-A1  *B-V*
.                    O       O       O
------------------------------
Den          B-A2
Juden        I-A2
in           I-A2
Europa       I-A2
hatte        O
Hitler       B-A0
bereits      O
Anfang       O
1939         O
die          B-A1
Vernichtung  I-A1
angedroht    *B-V*
.            O
------------------------------
Seit        O
September   O
1941        O
waren       O
sie         B-A0
gezwungen   O
,           O
den         B-A1
Judenstern  I-A1
zu          O
tragen      *B-V*
.           O
------------------------------
Auf                 B-A2    O       O
der                 I-A2    O       O
Wannseekonferenz    I-A2    O       O
im                  I-A2    O       O
Januar              I-A2    O       O
1942                I-A2    O       O
wurden              *B-V*   O       O
Zuständigkeiten     B-A1    B-A1    O
und                 I-A1    I-A1    O
Organisation        I-A1    I-A1    O
bezüglich           I-A1    I-A1    O
der                 I-A1    I-A1    O
[UNK]               I-A1    I-A1    O
Endlösung           I-A1    I-A1    O
der                 I-A1    I-A1    O
Judenfrage          I-A1    I-A1    O
[UNK]               I-A1    O       O
beschlossen         I-A1    *B-V*   O
,                   O       O       O
nachdem             O       O       O
das                 O       O       B-A1
Morden              O       O       I-A1
der                 O       O       I-A1
Einsatzgruppen      O       O       I-A1
der                 O       O       I-A1
Sicherheitspolizei  O       O       I-A1
und                 O       O       I-A1
des                 O       O       I-A1
SD                  O       O       I-A1
bereits             O       O       O
im                  O       O       O
Juli                O       O       O
1941                O       O       O
begonnen            O       O       *B-V*
hatte               O       O       O
.                   O       O       O
------------------------------
Nach                       O
der                        O
Deportation                O
in                         O
Ghettos                    O
wie                        O
Theresienstadt             O
oder                       O
das                        O
Warschauer                 O
Ghetto                     O
wurde                      O
die                        B-A1
Ermordung                  I-A1
der                        I-A1
Juden                      I-A1
im                         O
besetzten                  O
Osten                      O
Europas                    O
seit                       O
Herbst                     O
1941                       O
mit                        B-A2
Gaskammern                 I-A2
und                        I-A2
Verbrennungseinrichtungen  I-A2
auch                       I-A2
industriell                I-A2
betrieben                  *B-V*
.                          O
------------------------------
Neben               B-A0
Auschwitz           I-A0
-                   O
Birkenau            O
gehörten            *B-V*
im                  O
Rahmen              O
der                 O
[UNK]               O
Aktion              O
Reinhardt           O
[UNK]               O
zu                  B-A1
den                 I-A1
großen              I-A1
Vernichtungslagern  I-A1
Belzec              I-A1
,                   I-A1
Sobibor             I-A1
und                 I-A1
Treblinka           I-A1
.                   O
------------------------------
Bis          O
zum          O
Kriegsende   O
wurden       O
etwa         B-A1
sechs        I-A1
Millionen    I-A1
europäische  I-A1
Juden        I-A1
ermordet     *B-V*
,            O
darunter     O
über         O
drei         O
Millionen    O
polnische    O
Juden        O
.            O
==============================
Womit        B-A1
wurde        O
der          O
Beginn       O
des          O
zweiten      O
Weltkrieges  O
ausgelöst    *B-V*
?            O
```



## XQuAD

Task: Predict answer span in a context given the context and a question

### 1

```
Viele           O
Jahre           O
lang            O
herrschte       *B-V*
im              O
Sudan           O
ein             B-A0
islamistisches  I-A0
Regime          I-A0
unter           B-A1
der             I-A1
Führung         I-A1
von             I-A1
Hassan          I-A1
al              I-A1
-               I-A1
Turabi          I-A1
.               O
------------------------------
Seine       B-A1    O       O
Nationale   I-A1    O       O
Islamische  I-A1    O       O
Front       I-A1    O       O
gewann      *B-V*   O       O
erstmals    O       O       O
an          B-A0    O       O
Einfluss    I-A0    O       O
,           O       O       O
als         O       O       O
der         O       B-A0    B-A1
Machthaber  O       I-A0    I-A1
General     O       I-A0    I-A1
Gaafar      O       I-A0    I-A1
al          O       I-A0    I-A1
-           O       I-A0    I-A1
Nimeiry     O       I-A0    I-A1
1979        O       I-A0    I-A1
Mitglieder  O       I-A0    I-A1
einlud      O       *B-V*   I-A1
,           O       O       O
in          O       O       B-A3
seiner      O       O       I-A3
Regierung   O       O       I-A3
zu          O       O       O
dienen      O       O       *B-V*
.           O       O       O
------------------------------
Turabi           O
baute            *B-V*
eine             B-A1
starke           I-A1
wirtschaftliche  I-A1
Basis            I-A1
mit              I-A1
Geld             I-A1
aus              I-A1
ausländischen    I-A1
islamistischen   I-A1
Bankensystemen   I-A1
auf              O
,                O
insbesondere     B-A1
von              I-A1
solchen          I-A1
mit              I-A1
Verbindung       I-A1
zu               I-A1
Saudi            I-A1
-                O
Arabien          O
.                O
------------------------------
Er                B-A0    B-A0    O       O
rekrutierte       *B-V*   O       O       O
und               O       O       O       O
baute             O       *B-V*   O       O
auch              B-A2    O       O       O
ein               I-A2    B-A1    O       O
Kader             I-A2    I-A1    O       O
einflussreicher   I-A1    I-A1    O       O
Loyalisten        I-A1    I-A1    O       O
auf               O       O       O       O
,                 O       O       O       O
indem             O       O       O       O
er                O       O       B-A0    O
sympathisierende  O       O       B-A1    O
Studenten         O       O       I-A1    O
in                O       O       B-A2    O
die               O       O       I-A2    O
Universität       O       O       I-A2    O
und               O       O       I-A2    O
Militärakademie   O       O       I-A2    O
einbrachte        O       O       *B-V*   O
,                 O       O       O       O
während           O       O       O       O
er                O       O       O       B-A0
als               O       O       O       B-A2
Bildungsminister  O       O       O       I-A2
amtierte          O       O       O       *B-V*
.                 O       O       O       O
==============================
Welche        B-A1
Organisation  I-A1
hat           O
General       B-A0
Gaafar        I-A0
al            I-A0
-             I-A0
Nimeiry       I-A0
Mitglieder    I-A0
eingeladen    O
,             O
in            B-A3
seiner        I-A3
Regierung     I-A3
zu            O
dienen        *B-V*
?             O
```


### 2


```
In                     O       O
den                    O       O
meisten                O       O
Ländern                O       O
unterliegt             *B-V*   O
die                    B-A0    O
Apotheke               I-A0    O
der                    I-A0    O
Apothekengesetzgebung  I-A0    O
;                      O       O
mit                    O       O
Anforderungen          O       O
an                     O       O
Lagerbedingungen       O       O
,                      O       O
obligatorischen        O       O
Warnhinweisen          O       O
,                      O       O
Ausrüstung             O       O
usw                    O       O
.                      O       O
,                      O       O
die                    O       B-A2
in                     O       O
der                    O       O
Gesetzgebung           O       O
festgelegt             O       *B-V*
sind                   O       O
.                      O       O
------------------------------
War                 *B-V*   O       O       O       O
es                  B-A1    O       O       O       O
früher              O       O       O       O       O
einmal              O       O       O       O       O
so                  O       O       O       O       O
,                   O       O       O       O       O
dass                B-A1    O       O       O       O
Apotheker           I-A1    B-A0    B-A0    O       O
in                  I-A1    B-A1    O       O       O
der                 I-A1    I-A1    O       O       O
Apotheke            I-A1    I-A1    O       O       O
blieben             I-A1    *B-V*   O       O       O
,                   I-A1    O       O       O       O
um                  I-A1    O       O       O       O
Medikamente         I-A1    O       B-A1    O       O
zu                  I-A1    O       O       O       O
zusammenzustellen   I-A1    O       O       O       O
/                   O       O       O       O       O
dosieren            O       O       *B-V*   O       O
,                   O       O       O       O       O
so                  O       O       O       O       O
gibt                O       O       O       *B-V*   O
es                  O       O       O       B-A1    O
nun                 O       O       O       O       O
einen               O       O       O       B-A2    O
zunehmenden         O       O       O       I-A2    O
Trend               O       O       O       I-A2    O
zum                 O       O       O       I-A2    O
Einsatz             O       O       O       I-A2    O
ausgebildeter       O       O       O       I-A2    O
Apothekentechniker  O       O       O       I-A2    O
,                   O       O       O       O       O
während             O       O       O       O       O
der                 O       O       O       O       B-A0
Apotheker           O       O       O       O       I-A0
mehr                O       O       O       O       O
Zeit                O       O       O       O       O
für                 O       O       O       O       O
die                 O       O       O       O       O
Kommunikation       O       O       O       O       O
mit                 O       O       O       O       B-A1
Patienten           O       O       O       O       I-A1
aufbringt           O       O       O       O       *B-V*
.                   O       O       O       O       O
------------------------------
sind                 O       O
jetzt                O       O
mehr                 B-A2    O
und                  I-A2    O
mehr                 I-A2    O
auf                  I-A2    O
Automatisierung      I-A2    O
angewiesen           *B-V*   O
,                    O       O
die                  B-C-A2  B-A0
sie                  I-C-A1  B-A3
bei                  I-C-A1  B-A0
ihren                I-C-A1  I-A0
neuen                I-C-A1  I-A0
Aufgaben             I-C-A1  I-A0
in                   I-C-A2  I-A0
der                  I-C-A1  I-A0
Bearbeitung          I-C-A2  I-A0
von                  I-C-A2  I-A0
Patientenrezepten    I-C-A1  I-A0
und                  I-C-A1  I-A0
Fragen               I-C-A2  I-A0
der                  I-C-A1  I-A0
Patientensicherheit  I-C-A1  I-A0
unterstützen         I-C-A1  *B-V*
.                    O       O
==============================
Mit                 B-A1
welchen             I-A1
neuen               I-A1
Aufgaben            I-A1
befassen            *B-V*
sich                O
Apothekentechniker  B-A0
jetzt               O
?                   O
```

