Clustering
==========

Imports
-------

.. code:: ipython3

    from src.data.nordskog_data import get_data
    from src.data.preprocessing import DataPreprocessor
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

Clustering
----------

.. code:: ipython3

    def train_cluster(text_df, n_clusters):
        texts = text_df['text'].values
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(texts)
        model = KMeans(n_clusters=n_clusters, init='k-means++',
                       random_state=99)
    
        print(tfidf.shape)
        clf = model.fit(tfidf)
        text_df['cluster label'] = clf.labels_
        print(text_df.groupby(['cluster label'])['text'].count())
    
        return text_df

.. code:: ipython3

    train, test = get_data()
    train.dropna(inplace=True)
    preprocessor = DataPreprocessor(train)
    preprocessor.remove_paragraphs_over_65_words()
    preprocessor.remove_paragraphs_over_65_words()
    train = preprocessor.data.copy()
    cluster_df = train_cluster(train, 20)
    cluster_df


.. parsed-literal::

    (5412, 11077)
    

.. parsed-literal::

    C:\Users\Eirik\anaconda3\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    

.. parsed-literal::

    cluster label
    0      395
    1      199
    2     1022
    3      350
    4       90
    5      109
    6      135
    7      451
    8      439
    9      121
    10     250
    11     677
    12      86
    13     230
    14      39
    15      93
    16     173
    17     139
    18     388
    19      26
    Name: text, dtype: int64
    



.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>text</th>
          <th>label</th>
          <th>cluster label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Vålerenga - Rosenborg 2-3</td>
          <td>Ignore</td>
          <td>2</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Sam Johnson ga vertene ledelsen, men Jonathan ...</td>
          <td>Goal/Assist</td>
          <td>7</td>
        </tr>
        <tr>
          <th>2</th>
          <td>På et hjørnespark langt på overtid kom avgjøre...</td>
          <td>Goal/Assist</td>
          <td>7</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Ti minutter før pause scoret Sam Johnson sitt ...</td>
          <td>Goal/Assist</td>
          <td>0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Vålerenga holdt 1-0-ledelsen bare frem til sis...</td>
          <td>Goal/Assist</td>
          <td>0</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>5521</th>
          <td>– Mateo har sagt at han ønsker å dra. Jeg vil ...</td>
          <td>quote</td>
          <td>3</td>
        </tr>
        <tr>
          <th>5522</th>
          <td>– Her gjør han en miss. Han står midt i mål, o...</td>
          <td>quote</td>
          <td>3</td>
        </tr>
        <tr>
          <th>5523</th>
          <td>– Vi kan ta med masse positivt fra kampen, for...</td>
          <td>quote</td>
          <td>10</td>
        </tr>
        <tr>
          <th>5524</th>
          <td>Den tyske midtbanespilleren kom til Bayern Mün...</td>
          <td>Player details</td>
          <td>11</td>
        </tr>
        <tr>
          <th>5525</th>
          <td>Bendtner har vært i norsk fotball siden mars 2...</td>
          <td>Player details</td>
          <td>11</td>
        </tr>
      </tbody>
    </table>
    <p>5412 rows × 3 columns</p>
    </div>



Exploring the clusters
----------------------

.. code:: ipython3

    def cluster_generator():
        for cluster in range(0,20):
            yield print(f'Cluster {cluster} - {cluster_df[cluster_df["cluster label"] == cluster].shape[0]} samples \n'+'_'*100+'\n')
            for text in cluster_df[cluster_df['cluster label'] == cluster]['text'][:10]:
                yield print(f'{text} \n' )
    cluster_gen = cluster_generator()
    
    def next_in_cluster_generator():
        for _ in range(0,11):
            next(cluster_gen)

Cluster 0:
~~~~~~~~~~

Key words: Goal, Goal, Own-Goal, Chance, Goal, Chance, Goal, Ball
posession, Corner, Goal

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 0 - 395 samples 
    ____________________________________________________________________________________________________
    
    Ti minutter før pause scoret Sam Johnson sitt første mål siden midten av mai da han dro seg fint inn fra venstre og curlet ballen i det lengste hjørnet via Hedenstad. 
    
    Vålerenga holdt 1-0-ledelsen bare frem til siste spilleminutt av omgangen. Jonathan Levis halvvolley på utsiden av 16-meteren gikk via Bård Finne og overlistet keeper Adam Larsen Kwarasey på hjemmelaget. 
    
    Andreomgang var bare drøyt fire minutter gammel da Rosenborg hadde snudd kampen. Hedenstads hjørnespark ble slått hardt på første stolpe. Enar Jääger var uheldig og headet ballen i eget mål. 
    
    Bård Finne skapte kampens første store mulighet da han snappet ballen fra Vegard Eggen Hedenstad, dro seg inn i banen og fyrte av, men ballen gikk i stolpen. 
    
    I det 48. minutt klarte endelig Sarpsborg å bryte seg gjennom Odd-forsvaret. Heinz utnyttet Vegard Bergans balltap på egen banehalvdel og spilte videre til Thomassen. S08-kapteinen dro seg innover i boksen og hamret inn 1-1 i venstre
                            hjørne. 
    
    Kvarteret før slutt burde Birk Risa punktert oppgjøret for Odd. Etter et Ruud-frispark løp Risa seg fri og fikk stå helt alene, men fra fem meter skjøt han ballen utenfor mål. 
    
    Rossbachs redning av Heinz-forsøket var imponerende. Han måtte rygge bakover mot eget mål før han kastet seg bakover og fikk ballen unna. 
    
    I det 65. spilleminutt tok Juventus godt vare på en kontringsmulighet. Emre Can avanserte fremover og slapp ballen til Ronaldo, som fra litt skrått hold satte ballen i det lengste hjørnet til sitt andre mål for ettermiddagen. 
    
    Så fikk Ranheim endelig tak i ballen, og endret kampbildet totalt. 
    
    En utoverskrudd Ranheim-corner landet på lengste, der Eirik Valla Dønnem fikk et kne på ballen. Ballen spratt inn mot målet, der Helmersen, som fikk sjansen i Michael Karlsens skadefravær, ventet. 
    
    

Cluster 1
~~~~~~~~~

keywords: Quote/Feelings, Quote/Pleased, Quote/No regrets,
Quote/Expectations, Quote/Booking, Quote/Missed chances, Quote/Goal,
Quote/Chance, Quote/Belief, Quote/Pitch

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 1 - 199 samples 
    ____________________________________________________________________________________________________
    
    – Jeg var litt engstelig, med alt som skjedde etter at jeg forlot Real Madrid for å komme hit, men det er slik livet er, sier Ronaldo etter kampen ifølge AS. 
    
    – Men jeg er fornøyd. Jeg vet at jeg jobbet godt, og at målene ville komme. Jeg setter pris på lagkameratene mine, som har hjulpet meg mye med å tilpasse meg den italienske serien. 
    
    – Jeg har ikke angret ett sekund, sier Fellah til VG. 
    
    – Jeg visste at jeg kom til en klubb der det forventes titler. Og de fikk en spiller som forventer å vinne. Vi passer sammen, sier Jebali. 
    
    – Jeg tror først det er Kind Mikalsen som gjør forseelsen, så jeg gir han gult kort først, så det røde, men får raskt på øret at jeg bommer på både farge og spiller. Siden det er en åpenbar scoringsmulighet er det klink rødt, sier Hobber Nilsen. 
    
    – Hvis det var noen som skulle score i denne kampen her, så var det meg. Jeg hadde mange sjanser, og burde scoret to til, sier Nguen til VG. 
    
    – Jeg skjønte det med en gang jeg traff ballen med vrista at den kom til å gå inn. Det var et veldig viktig poeng, sier Nguen. 
    
    – Jeg så en åpning, og prøvde meg på lengste. Men så fikk én av dem en fot på ballen, sier Nguen, om den første sjansen. 
    
    – Man leser hele tida at ingen har troa, men jeg spiller her og har hundre prosent troen. Jeg spiller her og ser hvordan spillerne og trenerne jobber hver eneste dag. Jeg har troa helt fram til regnestykket sier at det er umulig å berge plassen, sier Fellah. 
    
    – Jeg tror vel ikke akkurat at banen ble bedre av det, kommenterte Knutsen. 
    
    

Cluster 2
~~~~~~~~~

Keywords: Score, Injury, Chance, Next game, Garbage, Score, Story
building, Quote, Game comment, Score

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 2 - 1022 samples 
    ____________________________________________________________________________________________________
    
    Vålerenga - Rosenborg 2-3 
    
    Enar Jääger og Mohammed Abu måtte for øvrig ut med skade for VIF. 
    
    I siste spilleminutt hadde Samúel Kari Fridjónsson en god mulighet på frispark, men Rosenborg-målvakt André Hansen fikk slått unna. 
    
    Sarpsborg innleder torsdag sitt gruppespilleventyr i europaligaen med bortekamp mot tyrkiske Besiktas i Istanbul. Der må sarpingene regne med at det blir skikkelig kok. 
    
    (©NTB) 
    
    Ranheim-Strømsgodset 1-1 
    
    Strømsgodset lå lenge an til å tape 0-1 borte for Ranheim søndag. 
    
    
                        – Dette ene poenget kan vise seg å bli svært viktig for dem.  
    
    Steffen Iversen etterlyser mer støtte til Marcus Pedersen på topp. 
    
    Start - Lillestrøm 3-0 
    
    

Cluster 3
~~~~~~~~~

Keywords: Expert comment, Quote/Expert, Quote/Expert, Quote/Expert,
Quote/Expert. Quote/Expert, Quote/Player, Quote/Player, Quote/Player,
Storytelling

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 3 - 350 samples 
    ____________________________________________________________________________________________________
    
    For til tross for et viktig bortepoeng, er sannheten at Strømsgodset nok en gang underpresterte, mener TV 2s fotballekspert Jesper Mathisen. 
    
    – Det er blytungt for Lillestrøm, som går på et tap, får Amundsen utvist og Erling Knudtzon går ut med skade, sier TV 2s fotballekspert Jesper Mathisen. 
    
    – Han har vist at han duger, sier TV 2s fotballekspert Jesper Mathisen. 
    
    – Adeleke Akinyemi har vært lovende i perioder i enkelte kamper, men målene har manglet. Han var nær sist. Han scoret mye i Europaliga-kvaliken for sin forrige klubb, så han har vist at han duger. Alle som er glade i Kristiansand, Sørlandet og Start håper han er målscoreren Start har trengt, sier TV 2s fotballekspert Jesper Mahtisen. 
    
    – Mannen som ble hentet for mange, mange millioner på overgangsvinduets siste dag har startet nedbetalingen. Det er en strålende prestasjon. Han kjørte karusell med en Lillestrøm-forsvarer. Marius Amundsen forsvant langt ut på den
                            glatte banen i Kristiansand, sa TV 2s ekspert Jesper Mathisen i FotballXtra-studio. 
    
    – Helt korrekt. Så lenge det er holding, så er det ikke et ærlig forsøk, sier TV 2s fotballekspert Jesper Mathisen. 
    
    Nå håper Fellah at returen til norsk fotball og Sandefjord kan få fart på karrieren igjen. For selv om han har blitt 29 år, er teknikeren klar på at han har mer å komme med som fotballspiller. 
    
    – Det er helt sikkert 10 ganger opp og ned med fly den siste tiden, sier matchvinneren. 
    
    – Det er derfor de kjøpte meg, sier tunisieren. Han virker helt rolig av oppstyret, mener han vet hva han kan. Han er ikke overrasket over noen ting. 
    
    Og med matchvinnere som Issam Jebali i stallen, er det ikke sikkert at den påstanden blir feil til slutt. 
    
    

Cluster 4
~~~~~~~~~

Keywords: Goal, Goal, Summary, Score, Score, Score, Summary, Summary,
Goal, Table

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 4 - 90 samples 
    ____________________________________________________________________________________________________
    
    Cardiff tok ledelsen på Stamford Bridge ved Sol Bamba, da han scoret på Chelsea-keeper Kepa fra kloss hold etter 17 minutters spill.  
    
    Hazard-hat trick sendte Chelsea over Liverpool 
    
    Cardiff yppet seg tidvis i andre omgang, men Chelsea-innbytter Willian skaffet først et straffespark – som Hazard satte sikkert i nettet. Og så fikk Willian kjempetreff fra distanse, og hamret inn 4–1. 
    
    Chelsea – Cardiff 4-1 (2-1) 
    
    Mål: 0-1 Sol Bamba (16), 1-1 Eden Hazard (37), 2-1 Hazard (43), 3-1 Hazard (str. 80), 4-1 Willian (83). 
    
    Newcastle – Arsenal 1-2 (0-0) 
    
    Eden Hazard ble kampens store spiller med tre målene for Chelsea. Like før slutt la Willian på til 4-0 med fantastisk treff, før Cardiff reduserte. 
    
    Det startet ikke så lovende for Chelsea, for etter et kvarters spill headet Sean Morrison på tvers innenfor feltet og Sol Bamba kriget inn 1-0-målet for Cardiff. 
    
    To minutter før pause la Giroud igjen et innlegg fra Pedro Rodríguez til Eden Hazard, som via en Cardiff-forsvarer satte inn 2-1 for Chelsea. 
    
    Briljant Hazard sendte Chelsea opp på tabelltopp med hat-trick 
    
    

Cluster 5
~~~~~~~~~

Keywords: Watch the game x 10

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 5 - 109 samples 
    ____________________________________________________________________________________________________
    
    Se Besiktas mot Sarpsborg på TV 2 Sport 1 og Sumo torsdag fra kl. 18.15. 
    
     Se Mesterligaen på TV 2 Sport 1 og TV 2 Sumo på tirsdag og onsdag.
    
     
    
    Se Wolverhampton - Burnley på TV 2 Sport Premium og TV 2 Sumo søndag klokken 14.30
     
    
    Se Champions League-godbiten mellom Liverpool mot Paris Saint-Germain på TV 2 Sport 1 og Sumo tirsdag kveld fra klokken 20.00. 
    
     Se Watford-Manchester United på TV 2 Sport Premium og TV 2 Sumo lørdag fra klokken 18.30.  
    
    Se Watford - Manchester United på Sumo eller TV 2 Sport Premium på lørdag fra klokken 18.30. 
    
    Se Liverpool mot PSG på TV 2 Sport 1 og Sumo tirsdag fra klokken 20.00.  
    
    Se Watford mot Manchester United på TV 2 Sport Premium og Sumo lørdag kveld fra kl. 18.00. Kampstart kl. 18.30. 
    
    Liverpool spiller sin generalprøve i toppkampen mot Tottenham lørdag. Kampen sendes på TV 2 Sport Premium og Sumo fra klokken 13.00. 
    
    Se Watford-Manchester United lørdag 18.30 på TV 2 Sumo og TV 2 Sport Premium!
     
    
    

Cluster 6
~~~~~~~~~

Keywords: Garbage, Score, Storytelling, Statistics, Chance, Score,
Score, Summary, Storytelling, Storytelling

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 6 - 135 samples 
    ____________________________________________________________________________________________________
    
    * 84 for Manchester United. 
    
    (Watford – Manchester United 1–2) Watford har hatt en glohet start på Premier League med fire seire av fire mulige. Mot Manchester United kom de derimot fort ned på jorden igjen.  
    
    For etter seire mot Brighton, Burnley, Crystal Palace og Tottenham hadde Watford vind i seilene før de tok imot Manchester United hjemme på Vicarage Road. 
    
    Målet var også Lukakus 20. på 39 seriekamper for Manchester United. Kun Ruud van Nistelrooy (26), Robin van Persie (32) og Dwight York (34) brukte færre kamper på å nå 20 seriemål for klubben, ifølge Opta.  
    
    Manchester City skapte enormt med sjanser mot Fulham hjemme i Manchester, det kunne blitt langt flere enn tre scoringer for Pep Guardolas mannskap. 
    
    Manchester C. – Fulham 3-0 (2-0) 
    
    Watford-Manchester United 1-2 
    
    Manchester United vant etter praktomgang: – Noe av det bedre jeg har sett av dem 
    
    Manchester United stoppet Watfords seiersrekke. Mye takket være en glimrende førsteomgang. 
    
    Mye takket være en svært god førsteomgang. Da imponerte Manchester United, for anledningen antrukket i pastellrosa drakter. 
    
    

Cluster 7
~~~~~~~~~

Keywords: Summary, Match winner, Match winner, Match winner, Match
winner, Goal, Injury, Summary, Goal, Summary

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 7 - 451 samples 
    ____________________________________________________________________________________________________
    
    Sam Johnson ga vertene ledelsen, men Jonathan Levi og et selvmål av Enar Jääger snudde kampen, før tidligere Brann-spiller Amin Nouri ordnet 2-2 med mål i sin andre strake (!) hjemmekamp. 
    
    På et hjørnespark langt på overtid kom avgjørelsen for Rosenborg da et hjørnespark havnet hos nysigneringen Jebali på bakerste stolpe. 
    
    Nesten fem minutter på overtid kom avgjørelsen da Jebali snek seg inn på bakerste stolpe på hjørnespark og med en velplassert volley styrte inn 3-2. 
    
    Avgjørelsen falt da Ruud bøyde et frispark over muren og i venstre kryss sju minutter etter pause. 
    
    Gleden på et regnfylt Sarpsborg stadion varte ikke lenge. Bare fire minutter senere prikket Ruud inn vinnermålet på frispark. Amin Askar hoppet desperat på streken, men kom ikke høyt nok opp. 
    
    Hjemmelagets Andreas Helmerens sendte vertene i ledelsen etter 28 minutter og trodde nok at han hadde blitt matchvinner. 
    
    Et skår i gleden for bursdagsbarnet Pellegrini var at Marko Arnautovic, som til tider terroriserte Everton-forsvaret, måtte forlate banen med en skade etter 65 minutter. Da hadde han nettopp scoret West Hams 3-1-mål.​ 
    
    Mot Lillestrøm kom den endelig da han på nydelig vis satte inn 1-0, og i andreomgang hadde han muligheter til å score enda flere. Kevin Kabran doblet fra straffemerket, før Mathias Bringaker la på 3-0 etter pause. 
    
    Allerede etter fem minutters spill tok Start ledelsen ved Akinyemi, som rundlurte Marius Amundsen og satte inn 1-0 for kristiansanderne. 
    
    Start tok en livsviktig 3-0-seier mot Lillestrøm, som måtte spille store deler av kampen med ti spillere etter utvisningen av Marius Amundsen etter 19 minutter. 
    
    

Cluster 8
~~~~~~~~~

Keywords: Goal, Quote/Coach, Quote/Player, Goal, Quote/Coach, Booking,
Booking, Summary, Summary, Statement

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 8 - 439 samples 
    ____________________________________________________________________________________________________
    
    Det varte imidlertid ikke lenge, for i det 58. spilleminutt havnet en retur fra André Hansen i beina på Amin Nouri, som banket inn 2-2 for Vålerenga. Dermed har høyrebacken scoret i to strake hjemmekamper. 
    
    – Bittert. Det er mye mer bittert enn å tape drittkamp der du blir rundspilt og ikke fortjener noe, men i dag var det glød i øynene på gutta. Det var tro, hardt arbeid og masse offensivt spill. Det var en herlig ramme med fantastiske
                            supportere, sier Vålerenga-trener Ronny Deila til TV 2. 
    
    – Det var en fantastisk avslutning. Vi var slitne og det var en vanskelig kamp, sier matchvinneren til TV Norge.​ 
    
    Kevin Kabran var sikker og satte inn 2-0 mot Lillestrøm med ti spillere. 
    
    – Det var en kamp som var utrolig spesiell. Det skjer utrolig mye det første kvarteret, og det som skjer der blir avgjørende for kamp. En marerittstart for vår del med målene, straffen, rødt kort og Erling Knudtzon skadet på bare ti minutter, sier LSK-trener Jörgen Lennartsson til TV 2. 
    
    Start fikk straffespark og Amundsen ble utvist, selv om det først var Simen Kind Mikalsen som feilaktig ble vist det røde kortet. 
    
    Douglas Costa svarte med å sette albuen i ansiktet på Di Francesco, før han skallet til Sassuolo-spilleren. Costa var heldig og fikk først kun gult kort av dommeren. 
    
    Det var Everton som hadde initiativet i kampen og presset på for scoring, men bakover lakk de som en sil. 
    
    Juventus' innbytter, som var livlig etter å ha kommet innpå i andreomgang, ble taket hardt av Federico Di Francesco i forkant av Sassuolos redusering til 1-2. 
    
    Det var en lettet Ronaldo som kunne juble for mål, men 33-åringen kunne og burde scoret enda flere i søndagens kamp. 
    
    

Cluster 9
~~~~~~~~~

Keywords: Table, Table, Table, Result, Table, Save, Table, Table,
Summary, Summary

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 9 - 121 samples 
    ____________________________________________________________________________________________________
    
    Rosenborg er tilbake på toppen av Eliteserien etter at Brann lånte den i et drøyt døgn etter 3-1 over Haugesund. 
    
    Rosenborg leder to poeng foran Brann. Vålerenga er nummer seks med fem poeng opp til Haugesund på tredjeplass. 
    
    Odd er litt i dytten om dagen. Telemarkingene er ubeseiret på sine sju siste seriekamper. Odds forrige tap kom mot Tromsø 1. juli. Dag-Eilev Fagermos menn står med 30 poeng. Det er to poeng mindre enn Sarpsborg. 
    
    Ranet med seg ett poeng i Ranheim: – De må passe seg 
    
    Strømsgodset ranet til seg uavgjort på overtid. Det er ett poeng som kan vise seg å bli livsviktig i kampen for å unngå nedrykk. 
    
    Så dukket Tokmac Nguen opp to minutter på overtid og reddet et svært viktig poeng for drammenserne. 
    
    Med Start-seier og Stabæk-poeng har Strømsgodset dermed plutselig begge bein godt plantet i bunnstriden, kun fire poeng over direkte nedrykk. 
    
    Juventus, som har vunnet ligaen de sju siste sesongene, topper Serie A med full pott etter fire runder. Napoli følger på andreplass med ni poeng etter like mange kamper. 
    
    En scoring fra tidligere Brann-back Amin Nouri og en misbrukt kjempesjanse fra RBKs Yann-Erik de Lanlay i sluttminuttene - og det så ut som Brann og Rosenborg skulle stå like i poeng før de åtte siste kampene. 
    
    RANHEIM (VG) (Ranheim-Strømsgodset 1–1) Med ett av de siste sparkene på ballen reddet Tokmac Nguen (24) poeng for Strømsgodset, og snudde glede til fortvilelse blant hjemmelagets spillere. 
    
    

Cluster 10
~~~~~~~~~~

Keywords:

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 10 - 250 samples 
    ____________________________________________________________________________________________________
    
    – Det er helt nifst å se dette Strømsgodset-laget. Med alle de gode spillerne vet vi hva de kan gjøre på sitt beste. Se laget, stallen og banken. At ikke de kan skape mål, sjanser og vinne kamper, er helt utrolig, sier Jesper Mathisen
                            i FotballXtra. 
    
    – Vi er ikke tre mål bedre enn Lillestrøm, men tilfeldighetene gjorde det sånn, sier Start-trener Kjetil Rekdal til Eurosport. 
    
    – Det er menneskelig å være slik. Av og til kan vi ikke kontrollere angsten vår. 
    
    – Siden mars måned intensiverte vi interessen for Jebali. Så slo vi til. Men folk sa det var «årets bom-kjøp», sier Bjørnebye - ironisk. 
    
    – Da vi fikk corneren på overtid, så trodde jeg på seier, sier Stig Inge Bjørnebye. Og corneren ekspederte Issam Jebali i mål. 
    
    – Fra halvspilt sesong og ut, spiller vi dobbelt så mange kamper som Brann. Vi har minimum 15 kamper, kanskje både 16 og 17, mer enn Brann siden juli og ut sesongen. Men vi ser på dette som en fordel, sier Bjørnebye. 
    
    – Seriemesterskapet skal vi vinne, hører vi Rini Coolen si. 
    
    – Spillerne er utrolig skuffet. Hva skal jeg kalle det? «Ransskuffet». Ikke over prestasjonen, for dette var opp imot det beste vi kan. Det så ut til å holde til tre poeng. Når vi først har seieren i lomma, to minutter på overtid, så skal vi greie å ri det unna, sier Ranheim-trener Svein Maalen, til VG. 
    
    – Kampen blir for oppjaga og vi har ballen altfor lite på slutten. Aller mest ligger det i at vi burde scoret 2-0 på en av de mange kontringsmulighetene vi hadde. Vi skal straffe dem mye hardere der, og ikke søle bort sjansene. 
    
    – Vi spilte en grei bortekamp, og hadde bra struktur defensivt. Vi har lett for å score mål, men slipper inn for mye mål. Jeg er godt fornøyd med ett poeng. 
    
    

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 11 - 677 samples 
    ____________________________________________________________________________________________________
    
    Drammenserne har ikke vunnet på fem kamper, og nå venter Molde, Sarpsborg 08 og Haugesund de neste tre rundene. 
    
    – De skal passe seg nå hvis de ikke klarer å heve seg. 
    
    – Selv om de har Marcus Pedersen med 14 mål allerede, trenger han mer støtte som spiss. De trenger at midtbanespillerne kommer opp og at ikke alt blir liggende på skuldrene til Marcus, sier Steffen Iversen. 
    
    Bjørn Petter Ingebretsen tok over som hovedtrener for Strømsgodset i sommer etter at Tor Ole Skullerud tok konsekvensen av en rekke dårlige resultater. Etter en brukbar start har imidlertid fremgangen uteblitt. 
    
    John Arne Riise var ukens gjest i FotballXtra. Han er bekymret for at drammenserne har trent for dårlig. 
    
    – De gjør for mange personlige feil og blir straffet hver gang. Jeg husker de spilte fantastisk fotball og det kom en bølge med løp. Alt går feil vei nå, og selvfølgelig har fått seg en knekk. Det skal du ikke kimse av, det har ekstremt
                            mye å si i fotball. 
    
    Adeleke Akinyemi ble hentet for om lag ti millioner kroner fra Ventspils i august, men scoringene har latt vente på seg. 
    
    – Jeg ble overrasket av reaksjonen hans. Vi har sluppet inn et mål, fått et rødt kort og en suspensjon vil bli langvarig. 
    
    De siste to årene, i Nordsjælland, har nemlig vært tunge. Fellah har knapt spilt kamper. Sist han startet en kamp før Sandefjord-returen, var 27. september 2017. 
    
    SANDEFJORD (VG) Han har slitt benken og følt seg glemt i hjemlandet. Som Sandefjord-spiller vil Mohamed Fellah (29) bevise for det norske folk at han fortsatt har det i seg. 
    
    

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 12 - 86 samples 
    ____________________________________________________________________________________________________
    
    Vinner selv på dårlige dager - se video under her:  
    
    Se hvordan Start senket LSK her: 
    
    • Her er søndagens oddstips 
    
     Her får du tilgang! ​ 
     
    
    Les minutt for minutt-referat fra kampen i TV 2s livesenter her. 
    
    Thomas Lehne Olsen er «offside-kongen». Se alle avgjørelsene her: 
    
    • Her er fredagens oddstips 
    
    Se video fra 3–0-seieren under her 
    
    Se begge intervjuene her og døm selv: 
    
    Flere «stikk» her - se video: 
    
    

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 13 - 230 samples 
    ____________________________________________________________________________________________________
    
    Ukraineren fikk æren av å sette ballen i det tomme nettet i sin første Premier League-start. 
    
    På overtid av første omgang fikk blåtrøyene sin redusering. Da hadde Marcos Silva nettopp gjort et offensivt bytte da Morgan Schneiderlin måtte vike vei for lille Bernard som ble hentet gratis i sommer. 
    
    Andreomgang startet like heseblesende som den første. Etter 60 minutter satt den igjen for bortelaget. Obiang og Arnautovic kombinerte nydelig enda en gang og åpnet opp Everton-forsvaret. 
    
    Cristiano Ronaldo måtte gå frustrert og målløs av banen i sine tre første kamper for Juventus etter sommerens overgang. 
    
    Ranheim så ut til å gå mot sin første seier på det nylagte kunstgresset. Men to minutter på overtid ødela en driblekonge fra Kenya moroa. 
    
    Odd spilte for første gang med 17-åringen Joshua Kitolano fra start. 
    
    33-åringen brukte 320 minutter på å score sitt første ligamål for Juventus, men i den fjerde ligakampen løsnet det: 
    
    Lenge måtte de faktisk jage for å komme à jour med baskerne, som tok ledelsen i første omgang.  
    
    Han vant den duellen han måtte vinne for å score sitt første mål i kamp fra start: 0–1 på Haugesund stadion, etter ni minutters spill. 
    
    Det var Kings tredje scoring for sesongen, han står også med en målgivende fra før. Med fem målpoeng på de første fem kampene, har den norske spissen fått en strålende start på Premier League-sesongen.  
    
    

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 14 - 39 samples 
    ____________________________________________________________________________________________________
    
    Bulgaria - Norge 1-0 
    
    Se 
    Bulgaria-Norge søndag 18.00 på TV 2!
    
     
    
    ​Se Bulgaria-Norge søndag 18.00 på TV 2! 
    
    Se Bulgaria-Norge på TV 2 og Sumo søndag fra kl. 17.30
     
    
    ​Se Bulgaria-Norge søndag 18.00 på TV 2! 
    
    Se Bulgaria-Norge søndag 18.00 på TV 2! 
    
    Se Bulgaria-Norge søndag 18.00 på TV 2!
     
    
    Norge-Kypros 2-0 
    
    Se Bulgaria-Norge søndag 18.00 på TV 2! 
    
    Se Bulgaria-Norge søndag 18.00 på TV 2! 
    
    

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 15 - 93 samples 
    ____________________________________________________________________________________________________
    
    I sin første sesong for Haugesund, på utlån fra den danske klubben, har nigerianeren vært den fremste offensive bidragsyteren på den nåværende tredjeplassen i Eliteserien. Seks mål og seks assists gjør ham til en av ligaens fremste målpoeng-plukkere – kun slått av Marcus Pedersen (15) og Erling Braut Håland (13). 
    
    Molde var det beste laget på Aker stadion i første omgang av den siste play off-kampen for Europaliga-gruppespillet. Tidlig i kampen kunne Erling Braut Haaland sende Molde i ledelsen ta han kom alene med keeper, men skuddet gikk midt på Zenit-keeperen.  
    
    MOLDE (VG) (Molde – Zenit 2–1, 3–4 sammenlagt) Molde kjempet og kjempet mot et godt organisert Zenit, men det ble bare nesten for Solskjær og hans menn. 
    
    Eirik Hestad og Erling Haaland scoret målene da Molde vendte til 2-1-seier på hjemmebane. Det tredje målet som ville gitt ekstraomganger kom aldri. 
    
    Haaland fullfører nemlig sesongen med Molde, før han offisielt blir RB Salzburg 1. januar 2019. Der blir han trolig værende en stund. Haaland har skrevet under på en femårskontrakt med den østerrikske storklubben. 
    
    Braut Haaland klar for RB Salzburg: – Vinn-vinn-situasjon for Molde 
    
    Moldenserne hadde store forhåpninger og forventninger om å sluttføre Haaland-avtalen med østerrikske Red Bull Salzburg i begynnelsen av uken, men da den norske overgangsfristen utløp natt til torsdag var overgangen fortsatt ikke bekreftet, er denne løsningen utelukket. 
    
    Zenit St. Petersburg-Molde 3-1 
    
    Braut Håland imponerte tross Molde-kollaps: – Landslagssjefen må elske det han ser 
    
    Erling Braut Håland markerte seg mot de meritterte Zenit-forsvarerne, med fart, kraft og teknisk frekke detaljer. 
    
    

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 16 - 173 samples 
    ____________________________________________________________________________________________________
    
    Se målene i Sportsnyhetene øverst. 
    
     Se begge målene i Sportsnyhetene øverst. 
    
     
    
     Se kampsammendraget i videovinduet øverst. 
    
     
    
     Se målene i Sportsnyhetene øverst!
     
    
     Se målene i Sportsnyhetene øverst!
     
    
    Se skrekkskaden i Sportsnyhetene øverst. 
    
     Se Lucas Moura herje med Manchester United i videovinduet øverst.
    
     
    
    Se reportasje fra Wolverhampton-museet i videovinduet øverst  
    
    Se målene i videovinduet øverst. 
    
     Se hvordan United slo Watford øverst!
     
    
    

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 17 - 139 samples 
    ____________________________________________________________________________________________________
    
    (Juventus-Sassuolo 2–1) Som i sin siste sesong for Real Madrid hadde Cristiano Ronaldo 27 forsøk før han endelig scoret for Juventus. 
    
    * 311 for Real Madrid. 
    
    Ronaldo gikk fra Real Madrid til Juventus i sommer. Pris: Snaut 950 millioner kroner. Han har skrevet en fireårskontrakt med en forventet årslønn på i underkant av 300 mill. kroner. 
    
    For da Gareth Bale kom seg til dødlinjen snaue tre minutter senere, og la inn i feltet, var det ingen Athletic-spillere som plukket opp den 175 cm høye midtbanejuvelen. Isco stanget inn 1–1 for Real Madrid.  
    
    Real Madrid tapte poeng og terreng til Barcelona 
    
    (Atheltic Bilbao – Real Madrid 1–1) Etter at Barcelona vant tidligere lørdag måtte Real Madrid svare borte mot Atheltic Bilbao. Det gikk ikke som planlagt.  
    
    For selv med Real Madrids «ferske» angrepstrio Gareth Bale, Karim Benzema og Marco Asensio på topp, slet de hvitkledde fra hovedstaden med å sette ballen i mål.  
    
    Iker Muniain utnyttet et uorganisert Real Madrid-forsvar og kranglet inn 1–0 til vertene etter 32 minutter etter en skikkelig lagscoring. 
    
    Fikk du med deg hva som skjedde under kampen mellom Castilla og Atletico Madrid B tidligere denne måneden? 
    
    Etter 61 minutter tok Real Madrid-trener Julen Lopetegui ut mannen som nylig ble kåret til Årets spiller i Europa, Luka Modric, og satte inn Isco.  
    
    

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 18 - 388 samples 
    ____________________________________________________________________________________________________
    
    Sarpsborgs poengfangst i Eliteserien har stoppet helt. Espen Ruud scoret begge Odds mål i 2-1-seieren i Østfold søndag. 
    
    Espen Ruuds frisparkfot hylles etter at han ble matchvinner mot Sarpsborg. 
    
    Endelig løsnet det for Ronaldo: Scoret to mål i Juventus-seier 
    
     Se høydepunktene fra kampen mot Start:  
    
    Lagets to neste bortekamper er mot gulljagende Brann og Rosenborg. 
    
    To mål av Espen Ruud - og Sarpsborg gikk på sitt fjerde strake tap 
    
    Odd lot Sarpsborg trille ball - og scoret selv mål i 1. omgang. 
    
    Etter 27 forsøk løsnet det: Ronaldo scoret to 
    
    I fjor scoret Hegerberg 31 mål på 20 ligakamper for Lyon - i tillegg til 15 mål i Champions League som klubben vant. Det gjør at hun er en av tre nominerte til FIFAs kåring av «årets spiller». Vinneren blir offentliggjort 24. september. 
    
    Lyon tok sin tredje strake seier i den franske toppdivisjonen søndag med 3–0 borte mot Guingamp. Hegerberg spilte i 73 minutter uten scoring. Eugenie Le Sommer satte alle tre målene for Lyon. 
    
    

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 19 - 26 samples 
    ____________________________________________________________________________________________________
    
    Tipster gir deg ferske oddstips hver dag! 
    
    Tipster gir deg ferske oddstips hver dag! 
    
    Tipster gir deg ferske oddstips hver dag! 
    
     Tipster gir deg ferske oddstips hver dag!  
    
    Tipster gir deg ferske oddstips hver dag!  
    
    Tipster gir deg ferske oddstips hver dag! 
    
    Tipster gir deg ferske oddstips hver dag! 
    
    Tipster gir deg ferske oddstips hver dag! 
    
    Tipster gir deg ferske oddstips hver dag! 
    
     Tipster gir deg ferske oddstips hver dag!  
    
    


