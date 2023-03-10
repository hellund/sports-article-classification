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
          <td>V??lerenga - Rosenborg 2-3</td>
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
          <td>P?? et hj??rnespark langt p?? overtid kom avgj??re...</td>
          <td>Goal/Assist</td>
          <td>7</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Ti minutter f??r pause scoret Sam Johnson sitt ...</td>
          <td>Goal/Assist</td>
          <td>0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>V??lerenga holdt 1-0-ledelsen bare frem til sis...</td>
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
          <td>??? Mateo har sagt at han ??nsker ?? dra. Jeg vil ...</td>
          <td>quote</td>
          <td>3</td>
        </tr>
        <tr>
          <th>5522</th>
          <td>??? Her gj??r han en miss. Han st??r midt i m??l, o...</td>
          <td>quote</td>
          <td>3</td>
        </tr>
        <tr>
          <th>5523</th>
          <td>??? Vi kan ta med masse positivt fra kampen, for...</td>
          <td>quote</td>
          <td>10</td>
        </tr>
        <tr>
          <th>5524</th>
          <td>Den tyske midtbanespilleren kom til Bayern M??n...</td>
          <td>Player details</td>
          <td>11</td>
        </tr>
        <tr>
          <th>5525</th>
          <td>Bendtner har v??rt i norsk fotball siden mars 2...</td>
          <td>Player details</td>
          <td>11</td>
        </tr>
      </tbody>
    </table>
    <p>5412 rows ?? 3 columns</p>
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
    
    Ti minutter f??r pause scoret Sam Johnson sitt f??rste m??l siden midten av mai da han dro seg fint inn fra venstre og curlet ballen i det lengste hj??rnet via Hedenstad. 
    
    V??lerenga holdt 1-0-ledelsen bare frem til siste spilleminutt av omgangen. Jonathan Levis halvvolley p?? utsiden av 16-meteren gikk via B??rd Finne og overlistet keeper Adam Larsen Kwarasey p?? hjemmelaget. 
    
    Andreomgang var bare dr??yt fire minutter gammel da Rosenborg hadde snudd kampen. Hedenstads hj??rnespark ble sl??tt hardt p?? f??rste stolpe. Enar J????ger var uheldig og headet ballen i eget m??l. 
    
    B??rd Finne skapte kampens f??rste store mulighet da han snappet ballen fra Vegard Eggen Hedenstad, dro seg inn i banen og fyrte av, men ballen gikk i stolpen. 
    
    I det 48. minutt klarte endelig Sarpsborg ?? bryte seg gjennom Odd-forsvaret. Heinz utnyttet Vegard Bergans balltap p?? egen banehalvdel og spilte videre til Thomassen. S08-kapteinen dro seg innover i boksen og hamret inn 1-1 i venstre
                            hj??rne. 
    
    Kvarteret f??r slutt burde Birk Risa punktert oppgj??ret for Odd. Etter et Ruud-frispark l??p Risa seg fri og fikk st?? helt alene, men fra fem meter skj??t han ballen utenfor m??l. 
    
    Rossbachs redning av Heinz-fors??ket var imponerende. Han m??tte rygge bakover mot eget m??l f??r han kastet seg bakover og fikk ballen unna. 
    
    I det 65. spilleminutt tok Juventus godt vare p?? en kontringsmulighet. Emre Can avanserte fremover og slapp ballen til Ronaldo, som fra litt skr??tt hold satte ballen i det lengste hj??rnet til sitt andre m??l for ettermiddagen. 
    
    S?? fikk Ranheim endelig tak i ballen, og endret kampbildet totalt. 
    
    En utoverskrudd Ranheim-corner landet p?? lengste, der Eirik Valla D??nnem fikk et kne p?? ballen. Ballen spratt inn mot m??let, der Helmersen, som fikk sjansen i Michael Karlsens skadefrav??r, ventet. 
    
    

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
    
    ??? Jeg var litt engstelig, med alt som skjedde etter at jeg forlot Real Madrid for ?? komme hit, men det er slik livet er, sier Ronaldo etter kampen if??lge AS. 
    
    ??? Men jeg er forn??yd. Jeg vet at jeg jobbet godt, og at m??lene ville komme. Jeg setter pris p?? lagkameratene mine, som har hjulpet meg mye med ?? tilpasse meg den italienske serien. 
    
    ??? Jeg har ikke angret ett sekund, sier Fellah til VG. 
    
    ??? Jeg visste at jeg kom til en klubb der det forventes titler. Og de fikk en spiller som forventer ?? vinne. Vi passer sammen, sier Jebali. 
    
    ??? Jeg tror f??rst det er Kind Mikalsen som gj??r forseelsen, s?? jeg gir han gult kort f??rst, s?? det r??de, men f??r raskt p?? ??ret at jeg bommer p?? b??de farge og spiller. Siden det er en ??penbar scoringsmulighet er det klink r??dt, sier Hobber Nilsen. 
    
    ??? Hvis det var noen som skulle score i denne kampen her, s?? var det meg. Jeg hadde mange sjanser, og burde scoret to til, sier Nguen til VG. 
    
    ??? Jeg skj??nte det med en gang jeg traff ballen med vrista at den kom til ?? g?? inn. Det var et veldig viktig poeng, sier Nguen. 
    
    ??? Jeg s?? en ??pning, og pr??vde meg p?? lengste. Men s?? fikk ??n av dem en fot p?? ballen, sier Nguen, om den f??rste sjansen. 
    
    ??? Man leser hele tida at ingen har troa, men jeg spiller her og har hundre prosent troen. Jeg spiller her og ser hvordan spillerne og trenerne jobber hver eneste dag. Jeg har troa helt fram til regnestykket sier at det er umulig ?? berge plassen, sier Fellah. 
    
    ??? Jeg tror vel ikke akkurat at banen ble bedre av det, kommenterte Knutsen. 
    
    

Cluster 2
~~~~~~~~~

Keywords: Score, Injury, Chance, Next game, Garbage, Score, Story
building, Quote, Game comment, Score

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 2 - 1022 samples 
    ____________________________________________________________________________________________________
    
    V??lerenga - Rosenborg 2-3 
    
    Enar J????ger og Mohammed Abu m??tte for ??vrig ut med skade for VIF. 
    
    I siste spilleminutt hadde Sam??el Kari Fridj??nsson en god mulighet p?? frispark, men Rosenborg-m??lvakt Andr?? Hansen fikk sl??tt unna. 
    
    Sarpsborg innleder torsdag sitt gruppespilleventyr i europaligaen med bortekamp mot tyrkiske Besiktas i Istanbul. Der m?? sarpingene regne med at det blir skikkelig kok. 
    
    (??NTB) 
    
    Ranheim-Str??msgodset 1-1 
    
    Str??msgodset l?? lenge an til ?? tape 0-1 borte for Ranheim s??ndag. 
    
    
                        ??? Dette ene poenget kan vise seg ?? bli sv??rt viktig for dem.  
    
    Steffen Iversen etterlyser mer st??tte til Marcus Pedersen p?? topp. 
    
    Start - Lillestr??m 3-0 
    
    

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
    
    For til tross for et viktig bortepoeng, er sannheten at Str??msgodset nok en gang underpresterte, mener TV 2s fotballekspert Jesper Mathisen. 
    
    ??? Det er blytungt for Lillestr??m, som g??r p?? et tap, f??r Amundsen utvist og Erling Knudtzon g??r ut med skade, sier TV 2s fotballekspert Jesper Mathisen. 
    
    ??? Han har vist at han duger, sier TV 2s fotballekspert Jesper Mathisen. 
    
    ??? Adeleke Akinyemi har v??rt lovende i perioder i enkelte kamper, men m??lene har manglet. Han var n??r sist. Han scoret mye i Europaliga-kvaliken for sin forrige klubb, s?? han har vist at han duger. Alle som er glade i Kristiansand, S??rlandet og Start h??per han er m??lscoreren Start har trengt, sier TV 2s fotballekspert Jesper Mahtisen. 
    
    ??? Mannen som ble hentet for mange, mange millioner p?? overgangsvinduets siste dag har startet nedbetalingen. Det er en str??lende prestasjon. Han kj??rte karusell med en Lillestr??m-forsvarer. Marius Amundsen forsvant langt ut p?? den
                            glatte banen i Kristiansand, sa TV 2s ekspert Jesper Mathisen i FotballXtra-studio. 
    
    ??? Helt korrekt. S?? lenge det er holding, s?? er det ikke et ??rlig fors??k, sier TV 2s fotballekspert Jesper Mathisen. 
    
    N?? h??per Fellah at returen til norsk fotball og Sandefjord kan f?? fart p?? karrieren igjen. For selv om han har blitt 29 ??r, er teknikeren klar p?? at han har mer ?? komme med som fotballspiller. 
    
    ??? Det er helt sikkert 10 ganger opp og ned med fly den siste tiden, sier matchvinneren. 
    
    ??? Det er derfor de kj??pte meg, sier tunisieren. Han virker helt rolig av oppstyret, mener han vet hva han kan. Han er ikke overrasket over noen ting. 
    
    Og med matchvinnere som Issam Jebali i stallen, er det ikke sikkert at den p??standen blir feil til slutt. 
    
    

Cluster 4
~~~~~~~~~

Keywords: Goal, Goal, Summary, Score, Score, Score, Summary, Summary,
Goal, Table

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 4 - 90 samples 
    ____________________________________________________________________________________________________
    
    Cardiff tok ledelsen p?? Stamford Bridge ved Sol Bamba, da han scoret p?? Chelsea-keeper Kepa fra kloss hold etter 17 minutters spill.  
    
    Hazard-hat trick sendte Chelsea over Liverpool 
    
    Cardiff yppet seg tidvis i andre omgang, men Chelsea-innbytter Willian skaffet f??rst et straffespark ??? som Hazard satte sikkert i nettet. Og s?? fikk Willian kjempetreff fra distanse, og hamret inn 4???1. 
    
    Chelsea ??? Cardiff 4-1 (2-1) 
    
    M??l: 0-1 Sol Bamba (16), 1-1 Eden Hazard (37), 2-1 Hazard (43), 3-1 Hazard (str. 80), 4-1 Willian (83). 
    
    Newcastle ??? Arsenal 1-2 (0-0) 
    
    Eden Hazard ble kampens store spiller med tre m??lene for Chelsea. Like f??r slutt la Willian p?? til 4-0 med fantastisk treff, f??r Cardiff reduserte. 
    
    Det startet ikke s?? lovende for Chelsea, for etter et kvarters spill headet Sean Morrison p?? tvers innenfor feltet og Sol Bamba kriget inn 1-0-m??let for Cardiff. 
    
    To minutter f??r pause la Giroud igjen et innlegg fra Pedro Rodr??guez til Eden Hazard, som via en Cardiff-forsvarer satte inn 2-1 for Chelsea. 
    
    Briljant Hazard sendte Chelsea opp p?? tabelltopp med hat-trick 
    
    

Cluster 5
~~~~~~~~~

Keywords: Watch the game x 10

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 5 - 109 samples 
    ____________________________________________________________________________________________________
    
    Se Besiktas mot Sarpsborg p?? TV 2 Sport 1 og Sumo torsdag fra kl. 18.15. 
    
     Se Mesterligaen p?? TV 2 Sport 1 og TV 2 Sumo p?? tirsdag og onsdag.
    
     
    
    Se Wolverhampton - Burnley p?? TV 2 Sport Premium og TV 2 Sumo s??ndag klokken 14.30
     
    
    Se Champions League-godbiten mellom Liverpool mot Paris Saint-Germain p?? TV 2 Sport 1 og Sumo tirsdag kveld fra klokken 20.00. 
    
     Se Watford-Manchester United p?? TV 2 Sport Premium og TV 2 Sumo l??rdag fra klokken 18.30.  
    
    Se Watford - Manchester United p?? Sumo eller TV 2 Sport Premium p?? l??rdag fra klokken 18.30. 
    
    Se Liverpool mot PSG p?? TV 2 Sport 1 og Sumo tirsdag fra klokken 20.00.  
    
    Se Watford mot Manchester United p?? TV 2 Sport Premium og Sumo l??rdag kveld fra kl. 18.00. Kampstart kl. 18.30. 
    
    Liverpool spiller sin generalpr??ve i toppkampen mot Tottenham l??rdag. Kampen sendes p?? TV 2 Sport Premium og Sumo fra klokken 13.00. 
    
    Se Watford-Manchester United l??rdag 18.30 p?? TV 2 Sumo og TV 2 Sport Premium!
     
    
    

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
    
    (Watford ??? Manchester United 1???2) Watford har hatt en glohet start p?? Premier League med fire seire av fire mulige. Mot Manchester United kom de derimot fort ned p?? jorden igjen.  
    
    For etter seire mot Brighton, Burnley, Crystal Palace og Tottenham hadde Watford vind i seilene f??r de tok imot Manchester United hjemme p?? Vicarage Road. 
    
    M??let var ogs?? Lukakus 20. p?? 39 seriekamper for Manchester United. Kun Ruud van Nistelrooy (26), Robin van Persie (32) og Dwight York (34) brukte f??rre kamper p?? ?? n?? 20 seriem??l for klubben, if??lge Opta.  
    
    Manchester City skapte enormt med sjanser mot Fulham hjemme i Manchester, det kunne blitt langt flere enn tre scoringer for Pep Guardolas mannskap. 
    
    Manchester C. ??? Fulham 3-0 (2-0) 
    
    Watford-Manchester United 1-2 
    
    Manchester United vant etter praktomgang: ??? Noe av det bedre jeg har sett av dem 
    
    Manchester United stoppet Watfords seiersrekke. Mye takket v??re en glimrende f??rsteomgang. 
    
    Mye takket v??re en sv??rt god f??rsteomgang. Da imponerte Manchester United, for anledningen antrukket i pastellrosa drakter. 
    
    

Cluster 7
~~~~~~~~~

Keywords: Summary, Match winner, Match winner, Match winner, Match
winner, Goal, Injury, Summary, Goal, Summary

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 7 - 451 samples 
    ____________________________________________________________________________________________________
    
    Sam Johnson ga vertene ledelsen, men Jonathan Levi og et selvm??l av Enar J????ger snudde kampen, f??r tidligere Brann-spiller Amin Nouri ordnet 2-2 med m??l i sin andre strake (!) hjemmekamp. 
    
    P?? et hj??rnespark langt p?? overtid kom avgj??relsen for Rosenborg da et hj??rnespark havnet hos nysigneringen Jebali p?? bakerste stolpe. 
    
    Nesten fem minutter p?? overtid kom avgj??relsen da Jebali snek seg inn p?? bakerste stolpe p?? hj??rnespark og med en velplassert volley styrte inn 3-2. 
    
    Avgj??relsen falt da Ruud b??yde et frispark over muren og i venstre kryss sju minutter etter pause. 
    
    Gleden p?? et regnfylt Sarpsborg stadion varte ikke lenge. Bare fire minutter senere prikket Ruud inn vinnerm??let p?? frispark. Amin Askar hoppet desperat p?? streken, men kom ikke h??yt nok opp. 
    
    Hjemmelagets Andreas Helmerens sendte vertene i ledelsen etter 28 minutter og trodde nok at han hadde blitt matchvinner. 
    
    Et sk??r i gleden for bursdagsbarnet Pellegrini var at Marko Arnautovic, som til tider terroriserte Everton-forsvaret, m??tte forlate banen med en skade etter 65 minutter. Da hadde han nettopp scoret West Hams 3-1-m??l.??? 
    
    Mot Lillestr??m kom den endelig da han p?? nydelig vis satte inn 1-0, og i andreomgang hadde han muligheter til ?? score enda flere. Kevin Kabran doblet fra straffemerket, f??r Mathias Bringaker la p?? 3-0 etter pause. 
    
    Allerede etter fem minutters spill tok Start ledelsen ved Akinyemi, som rundlurte Marius Amundsen og satte inn 1-0 for kristiansanderne. 
    
    Start tok en livsviktig 3-0-seier mot Lillestr??m, som m??tte spille store deler av kampen med ti spillere etter utvisningen av Marius Amundsen etter 19 minutter. 
    
    

Cluster 8
~~~~~~~~~

Keywords: Goal, Quote/Coach, Quote/Player, Goal, Quote/Coach, Booking,
Booking, Summary, Summary, Statement

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 8 - 439 samples 
    ____________________________________________________________________________________________________
    
    Det varte imidlertid ikke lenge, for i det 58. spilleminutt havnet en retur fra Andr?? Hansen i beina p?? Amin Nouri, som banket inn 2-2 for V??lerenga. Dermed har h??yrebacken scoret i to strake hjemmekamper. 
    
    ??? Bittert. Det er mye mer bittert enn ?? tape drittkamp der du blir rundspilt og ikke fortjener noe, men i dag var det gl??d i ??ynene p?? gutta. Det var tro, hardt arbeid og masse offensivt spill. Det var en herlig ramme med fantastiske
                            supportere, sier V??lerenga-trener Ronny Deila til TV 2. 
    
    ??? Det var en fantastisk avslutning. Vi var slitne og det var en vanskelig kamp, sier matchvinneren til TV Norge.??? 
    
    Kevin Kabran var sikker og satte inn 2-0 mot Lillestr??m med ti spillere. 
    
    ??? Det var en kamp som var utrolig spesiell. Det skjer utrolig mye det f??rste kvarteret, og det som skjer der blir avgj??rende for kamp. En marerittstart for v??r del med m??lene, straffen, r??dt kort og Erling Knudtzon skadet p?? bare ti minutter, sier LSK-trener J??rgen Lennartsson til TV 2. 
    
    Start fikk straffespark og Amundsen ble utvist, selv om det f??rst var Simen Kind Mikalsen som feilaktig ble vist det r??de kortet. 
    
    Douglas Costa svarte med ?? sette albuen i ansiktet p?? Di Francesco, f??r han skallet til Sassuolo-spilleren. Costa var heldig og fikk f??rst kun gult kort av dommeren. 
    
    Det var Everton som hadde initiativet i kampen og presset p?? for scoring, men bakover lakk de som en sil. 
    
    Juventus' innbytter, som var livlig etter ?? ha kommet innp?? i andreomgang, ble taket hardt av Federico Di Francesco i forkant av Sassuolos redusering til 1-2. 
    
    Det var en lettet Ronaldo som kunne juble for m??l, men 33-??ringen kunne og burde scoret enda flere i s??ndagens kamp. 
    
    

Cluster 9
~~~~~~~~~

Keywords: Table, Table, Table, Result, Table, Save, Table, Table,
Summary, Summary

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 9 - 121 samples 
    ____________________________________________________________________________________________________
    
    Rosenborg er tilbake p?? toppen av Eliteserien etter at Brann l??nte den i et dr??yt d??gn etter 3-1 over Haugesund. 
    
    Rosenborg leder to poeng foran Brann. V??lerenga er nummer seks med fem poeng opp til Haugesund p?? tredjeplass. 
    
    Odd er litt i dytten om dagen. Telemarkingene er ubeseiret p?? sine sju siste seriekamper. Odds forrige tap kom mot Troms?? 1. juli. Dag-Eilev Fagermos menn st??r med 30 poeng. Det er to poeng mindre enn Sarpsborg. 
    
    Ranet med seg ett poeng i Ranheim: ??? De m?? passe seg 
    
    Str??msgodset ranet til seg uavgjort p?? overtid. Det er ett poeng som kan vise seg ?? bli livsviktig i kampen for ?? unng?? nedrykk. 
    
    S?? dukket Tokmac Nguen opp to minutter p?? overtid og reddet et sv??rt viktig poeng for drammenserne. 
    
    Med Start-seier og Stab??k-poeng har Str??msgodset dermed plutselig begge bein godt plantet i bunnstriden, kun fire poeng over direkte nedrykk. 
    
    Juventus, som har vunnet ligaen de sju siste sesongene, topper Serie A med full pott etter fire runder. Napoli f??lger p?? andreplass med ni poeng etter like mange kamper. 
    
    En scoring fra tidligere Brann-back Amin Nouri og en misbrukt kjempesjanse fra RBKs Yann-Erik de Lanlay i sluttminuttene - og det s?? ut som Brann og Rosenborg skulle st?? like i poeng f??r de ??tte siste kampene. 
    
    RANHEIM (VG) (Ranheim-Str??msgodset 1???1) Med ett av de siste sparkene p?? ballen reddet Tokmac Nguen (24) poeng for Str??msgodset, og snudde glede til fortvilelse blant hjemmelagets spillere. 
    
    

Cluster 10
~~~~~~~~~~

Keywords: Quote/Expert, Quote/Coach, Quote/Feelings, Quote/Transfer,
Quote/Player, Quote, Quote/Coach, Quote/Summary, Quote

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 10 - 250 samples 
    ____________________________________________________________________________________________________
    
    ??? Det er helt nifst ?? se dette Str??msgodset-laget. Med alle de gode spillerne vet vi hva de kan gj??re p?? sitt beste. Se laget, stallen og banken. At ikke de kan skape m??l, sjanser og vinne kamper, er helt utrolig, sier Jesper Mathisen
                            i FotballXtra. 
    
    ??? Vi er ikke tre m??l bedre enn Lillestr??m, men tilfeldighetene gjorde det s??nn, sier Start-trener Kjetil Rekdal til Eurosport. 
    
    ??? Det er menneskelig ?? v??re slik. Av og til kan vi ikke kontrollere angsten v??r. 
    
    ??? Siden mars m??ned intensiverte vi interessen for Jebali. S?? slo vi til. Men folk sa det var ????rets bom-kj??p??, sier Bj??rnebye - ironisk. 
    
    ??? Da vi fikk corneren p?? overtid, s?? trodde jeg p?? seier, sier Stig Inge Bj??rnebye. Og corneren ekspederte Issam Jebali i m??l. 
    
    ??? Fra halvspilt sesong og ut, spiller vi dobbelt s?? mange kamper som Brann. Vi har minimum 15 kamper, kanskje b??de 16 og 17, mer enn Brann siden juli og ut sesongen. Men vi ser p?? dette som en fordel, sier Bj??rnebye. 
    
    ??? Seriemesterskapet skal vi vinne, h??rer vi Rini Coolen si. 
    
    ??? Spillerne er utrolig skuffet. Hva skal jeg kalle det? ??Ransskuffet??. Ikke over prestasjonen, for dette var opp imot det beste vi kan. Det s?? ut til ?? holde til tre poeng. N??r vi f??rst har seieren i lomma, to minutter p?? overtid, s?? skal vi greie ?? ri det unna, sier Ranheim-trener Svein Maalen, til VG. 
    
    ??? Kampen blir for oppjaga og vi har ballen altfor lite p?? slutten. Aller mest ligger det i at vi burde scoret 2-0 p?? en av de mange kontringsmulighetene vi hadde. Vi skal straffe dem mye hardere der, og ikke s??le bort sjansene. 
    
    ??? Vi spilte en grei bortekamp, og hadde bra struktur defensivt. Vi har lett for ?? score m??l, men slipper inn for mye m??l. Jeg er godt forn??yd med ett poeng. 
    
    

Cluster 11
~~~~~~~~~~

Keywords: Next game, Quote, Quote/Expert, Storytelling, Statement,
Quote, Transfer/Storytelling, Storytelling, Storytelling

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 11 - 677 samples 
    ____________________________________________________________________________________________________
    
    Drammenserne har ikke vunnet p?? fem kamper, og n?? venter Molde, Sarpsborg 08 og Haugesund de neste tre rundene. 
    
    ??? De skal passe seg n?? hvis de ikke klarer ?? heve seg. 
    
    ??? Selv om de har Marcus Pedersen med 14 m??l allerede, trenger han mer st??tte som spiss. De trenger at midtbanespillerne kommer opp og at ikke alt blir liggende p?? skuldrene til Marcus, sier Steffen Iversen. 
    
    Bj??rn Petter Ingebretsen tok over som hovedtrener for Str??msgodset i sommer etter at Tor Ole Skullerud tok konsekvensen av en rekke d??rlige resultater. Etter en brukbar start har imidlertid fremgangen uteblitt. 
    
    John Arne Riise var ukens gjest i FotballXtra. Han er bekymret for at drammenserne har trent for d??rlig. 
    
    ??? De gj??r for mange personlige feil og blir straffet hver gang. Jeg husker de spilte fantastisk fotball og det kom en b??lge med l??p. Alt g??r feil vei n??, og selvf??lgelig har f??tt seg en knekk. Det skal du ikke kimse av, det har ekstremt
                            mye ?? si i fotball. 
    
    Adeleke Akinyemi ble hentet for om lag ti millioner kroner fra Ventspils i august, men scoringene har latt vente p?? seg. 
    
    ??? Jeg ble overrasket av reaksjonen hans. Vi har sluppet inn et m??l, f??tt et r??dt kort og en suspensjon vil bli langvarig. 
    
    De siste to ??rene, i Nordsj??lland, har nemlig v??rt tunge. Fellah har knapt spilt kamper. Sist han startet en kamp f??r Sandefjord-returen, var 27. september 2017. 
    
    SANDEFJORD (VG) Han har slitt benken og f??lt seg glemt i hjemlandet. Som Sandefjord-spiller vil Mohamed Fellah (29) bevise for det norske folk at han fortsatt har det i seg. 
    
    

Cluster 12
~~~~~~~~~~

Keywords: Link x 10

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 12 - 86 samples 
    ____________________________________________________________________________________________________
    
    Vinner selv p?? d??rlige dager - se video under her:  
    
    Se hvordan Start senket LSK her: 
    
    ??? Her er s??ndagens oddstips 
    
     Her f??r du tilgang! ??? 
     
    
    Les minutt for minutt-referat fra kampen i TV 2s livesenter her. 
    
    Thomas Lehne Olsen er ??offside-kongen??. Se alle avgj??relsene her: 
    
    ??? Her er fredagens oddstips 
    
    Se video fra 3???0-seieren under her 
    
    Se begge intervjuene her og d??m selv: 
    
    Flere ??stikk?? her - se video: 
    
    

CLuster 13
~~~~~~~~~~

Keywords: Goal, Goal/Substitution, Goal, Player/Storytelling,
Storytelling/Goal, Debut, Goal, Storytelling, Goal, Gotal/Player

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 13 - 230 samples 
    ____________________________________________________________________________________________________
    
    Ukraineren fikk ??ren av ?? sette ballen i det tomme nettet i sin f??rste Premier League-start. 
    
    P?? overtid av f??rste omgang fikk bl??tr??yene sin redusering. Da hadde Marcos Silva nettopp gjort et offensivt bytte da Morgan Schneiderlin m??tte vike vei for lille Bernard som ble hentet gratis i sommer. 
    
    Andreomgang startet like heseblesende som den f??rste. Etter 60 minutter satt den igjen for bortelaget. Obiang og Arnautovic kombinerte nydelig enda en gang og ??pnet opp Everton-forsvaret. 
    
    Cristiano Ronaldo m??tte g?? frustrert og m??ll??s av banen i sine tre f??rste kamper for Juventus etter sommerens overgang. 
    
    Ranheim s?? ut til ?? g?? mot sin f??rste seier p?? det nylagte kunstgresset. Men to minutter p?? overtid ??dela en driblekonge fra Kenya moroa. 
    
    Odd spilte for f??rste gang med 17-??ringen Joshua Kitolano fra start. 
    
    33-??ringen brukte 320 minutter p?? ?? score sitt f??rste ligam??l for Juventus, men i den fjerde ligakampen l??snet det: 
    
    Lenge m??tte de faktisk jage for ?? komme ?? jour med baskerne, som tok ledelsen i f??rste omgang.  
    
    Han vant den duellen han m??tte vinne for ?? score sitt f??rste m??l i kamp fra start: 0???1 p?? Haugesund stadion, etter ni minutters spill. 
    
    Det var Kings tredje scoring for sesongen, han st??r ogs?? med en m??lgivende fra f??r. Med fem m??lpoeng p?? de f??rste fem kampene, har den norske spissen f??tt en str??lende start p?? Premier League-sesongen.  
    
    

Cluster 14
~~~~~~~~~~

Keywords: Watch the game x 10

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 14 - 39 samples 
    ____________________________________________________________________________________________________
    
    Bulgaria - Norge 1-0 
    
    Se 
    Bulgaria-Norge s??ndag 18.00 p?? TV 2!
    
     
    
    ???Se Bulgaria-Norge s??ndag 18.00 p?? TV 2! 
    
    Se Bulgaria-Norge p?? TV 2 og Sumo s??ndag fra kl. 17.30
     
    
    ???Se Bulgaria-Norge s??ndag 18.00 p?? TV 2! 
    
    Se Bulgaria-Norge s??ndag 18.00 p?? TV 2! 
    
    Se Bulgaria-Norge s??ndag 18.00 p?? TV 2!
     
    
    Norge-Kypros 2-0 
    
    Se Bulgaria-Norge s??ndag 18.00 p?? TV 2! 
    
    Se Bulgaria-Norge s??ndag 18.00 p?? TV 2! 
    
    

Cluster 15
~~~~~~~~~~

Storytelling/Player, Commentary/Chance, Summary, Goal/Storytelling,
Transfer/Storytelling, Transfer, Score, Statement, Statement

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 15 - 93 samples 
    ____________________________________________________________________________________________________
    
    I sin f??rste sesong for Haugesund, p?? utl??n fra den danske klubben, har nigerianeren v??rt den fremste offensive bidragsyteren p?? den n??v??rende tredjeplassen i Eliteserien. Seks m??l og seks assists gj??r ham til en av ligaens fremste m??lpoeng-plukkere ??? kun sl??tt av Marcus Pedersen (15) og Erling Braut H??land (13). 
    
    Molde var det beste laget p?? Aker stadion i f??rste omgang av den siste play off-kampen for Europaliga-gruppespillet. Tidlig i kampen kunne Erling Braut Haaland sende Molde i ledelsen ta han kom alene med keeper, men skuddet gikk midt p?? Zenit-keeperen.  
    
    MOLDE (VG) (Molde ??? Zenit 2???1, 3???4 sammenlagt) Molde kjempet og kjempet mot et godt organisert Zenit, men det ble bare nesten for Solskj??r og hans menn. 
    
    Eirik Hestad og Erling Haaland scoret m??lene da Molde vendte til 2-1-seier p?? hjemmebane. Det tredje m??let som ville gitt ekstraomganger kom aldri. 
    
    Haaland fullf??rer nemlig sesongen med Molde, f??r han offisielt blir RB Salzburg 1. januar 2019. Der blir han trolig v??rende en stund. Haaland har skrevet under p?? en fem??rskontrakt med den ??sterrikske storklubben. 
    
    Braut Haaland klar for RB Salzburg: ??? Vinn-vinn-situasjon for Molde 
    
    Moldenserne hadde store forh??pninger og forventninger om ?? sluttf??re Haaland-avtalen med ??sterrikske Red Bull Salzburg i begynnelsen av uken, men da den norske overgangsfristen utl??p natt til torsdag var overgangen fortsatt ikke bekreftet, er denne l??sningen utelukket. 
    
    Zenit St. Petersburg-Molde 3-1 
    
    Braut H??land imponerte tross Molde-kollaps: ??? Landslagssjefen m?? elske det han ser 
    
    Erling Braut H??land markerte seg mot de meritterte Zenit-forsvarerne, med fart, kraft og teknisk frekke detaljer. 
    
    

Cluster 16
~~~~~~~~~~

Keywords: Watch the goals x 10

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 16 - 173 samples 
    ____________________________________________________________________________________________________
    
    Se m??lene i Sportsnyhetene ??verst. 
    
     Se begge m??lene i Sportsnyhetene ??verst. 
    
     
    
     Se kampsammendraget i videovinduet ??verst. 
    
     
    
     Se m??lene i Sportsnyhetene ??verst!
     
    
     Se m??lene i Sportsnyhetene ??verst!
     
    
    Se skrekkskaden i Sportsnyhetene ??verst. 
    
     Se Lucas Moura herje med Manchester United i videovinduet ??verst.
    
     
    
    Se reportasje fra Wolverhampton-museet i videovinduet ??verst  
    
    Se m??lene i videovinduet ??verst. 
    
     Se hvordan United slo Watford ??verst!
     
    
    

Cluster 17
~~~~~~~~~~

Real madrid x 10

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 17 - 139 samples 
    ____________________________________________________________________________________________________
    
    (Juventus-Sassuolo 2???1) Som i sin siste sesong for Real Madrid hadde Cristiano Ronaldo 27 fors??k f??r han endelig scoret for Juventus. 
    
    * 311 for Real Madrid. 
    
    Ronaldo gikk fra Real Madrid til Juventus i sommer. Pris: Snaut 950 millioner kroner. Han har skrevet en fire??rskontrakt med en forventet ??rsl??nn p?? i underkant av 300 mill. kroner. 
    
    For da Gareth Bale kom seg til d??dlinjen snaue tre minutter senere, og la inn i feltet, var det ingen Athletic-spillere som plukket opp den 175 cm h??ye midtbanejuvelen. Isco stanget inn 1???1 for Real Madrid.  
    
    Real Madrid tapte poeng og terreng til Barcelona 
    
    (Atheltic Bilbao ??? Real Madrid 1???1) Etter at Barcelona vant tidligere l??rdag m??tte Real Madrid svare borte mot Atheltic Bilbao. Det gikk ikke som planlagt.  
    
    For selv med Real Madrids ??ferske?? angrepstrio Gareth Bale, Karim Benzema og Marco Asensio p?? topp, slet de hvitkledde fra hovedstaden med ?? sette ballen i m??l.  
    
    Iker Muniain utnyttet et uorganisert Real Madrid-forsvar og kranglet inn 1???0 til vertene etter 32 minutter etter en skikkelig lagscoring. 
    
    Fikk du med deg hva som skjedde under kampen mellom Castilla og Atletico Madrid B tidligere denne m??neden? 
    
    Etter 61 minutter tok Real Madrid-trener Julen Lopetegui ut mannen som nylig ble k??ret til ??rets spiller i Europa, Luka Modric, og satte inn Isco.  
    
    

Cluster 18
~~~~~~~~~~

Keywords: Matchwinner/Table, Matchwinner, Goal/Summary, Watch higlights,
Next games, Summary, Summary, Summary/Goal, Summary/Storytelling/Player,
Summary/Goal

.. code:: ipython3

    next_in_cluster_generator()


.. parsed-literal::

    Cluster 18 - 388 samples 
    ____________________________________________________________________________________________________
    
    Sarpsborgs poengfangst i Eliteserien har stoppet helt. Espen Ruud scoret begge Odds m??l i 2-1-seieren i ??stfold s??ndag. 
    
    Espen Ruuds frisparkfot hylles etter at han ble matchvinner mot Sarpsborg. 
    
    Endelig l??snet det for Ronaldo: Scoret to m??l i Juventus-seier 
    
     Se h??ydepunktene fra kampen mot Start:  
    
    Lagets to neste bortekamper er mot gulljagende Brann og Rosenborg. 
    
    To m??l av Espen Ruud - og Sarpsborg gikk p?? sitt fjerde strake tap 
    
    Odd lot Sarpsborg trille ball - og scoret selv m??l i 1. omgang. 
    
    Etter 27 fors??k l??snet det: Ronaldo scoret to 
    
    I fjor scoret Hegerberg 31 m??l p?? 20 ligakamper for Lyon - i tillegg til 15 m??l i Champions League som klubben vant. Det gj??r at hun er en av tre nominerte til FIFAs k??ring av ????rets spiller??. Vinneren blir offentliggjort 24. september. 
    
    Lyon tok sin tredje strake seier i den franske toppdivisjonen s??ndag med 3???0 borte mot Guingamp. Hegerberg spilte i 73 minutter uten scoring. Eugenie Le Sommer satte alle tre m??lene for Lyon. 
    
    

Cluster 19
~~~~~~~~~~

Keywords: Odds x 10

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
    
    


