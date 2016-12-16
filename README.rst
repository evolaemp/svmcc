=====
svmcc
=====

This repository accompanies the paper "Using support vector machines and
state-of-the-art algorithms for phonetic alignment to identify cognates in
multi-lingual wordlists" by Jäger, List and Sofroniev. The repository contains
both the data and the source code used in the paper's experiment.


data
====

datasets
--------

+--------------------+------------------------------+------------------------------+
| Dataset            | Language Families            | Source                       |
+====================+==============================+==============================+
| abvd               | Austronesian                 | Greenhill et al, 2008        |
+--------------------+------------------------------+------------------------------+
| afrasian           | Afro-Asiatic                 | Militarev, 2000              |
+--------------------+------------------------------+------------------------------+
| bai                | Sino-Tibetan                 | Wang, 2006                   |
+--------------------+------------------------------+------------------------------+
| central_asian      | Turkic, Indo-European        | Manni et al, 2016            |
+--------------------+------------------------------+------------------------------+
| chinese_2004       | Sino-Tibetan                 | Hóu, 2004                    |
+--------------------+------------------------------+------------------------------+
| chinese_1964       | Sino-Tibetan                 | Běijīng Dàxué, 1964          |
+--------------------+------------------------------+------------------------------+
| huon               | Trans-New Guinea             | McElhanon, 1967              |
+--------------------+------------------------------+------------------------------+
| ielex              | Indo-European                | Dunn, 2012                   |
+--------------------+------------------------------+------------------------------+
| japanese           | Japonic                      | Hattori, 1973                |
+--------------------+------------------------------+------------------------------+
| kadai              | Tai-Kadai                    | Peiros, 1998                 |
+--------------------+------------------------------+------------------------------+
| kamasau            | Torricelli                   | Sanders, 1980                |
+--------------------+------------------------------+------------------------------+
| lolo_burmese       | Sino-Tibetan                 | Peiros, 1998                 |
+--------------------+------------------------------+------------------------------+
| mayan              | Mayan                        | Brown, 2008                  |
+--------------------+------------------------------+------------------------------+
| miao_yao           | Hmong-Mien                   | Peiros, 1998                 |
+--------------------+------------------------------+------------------------------+
| mixe_zoque         | Mixe-Zoque                   | Cysouw et al, 2006           |
+--------------------+------------------------------+------------------------------+
| mon_khmer          | Austroasiatic                | Peiros, 1998                 |
+--------------------+------------------------------+------------------------------+
| ob_ugrian          | Uralic                       | Zhivlov, 2011                |
+--------------------+------------------------------+------------------------------+
| tujia              | Sino-Tibetan                 | Starostin, 2013              |
+--------------------+------------------------------+------------------------------+

Each dataset is stored in a `tsv`_ file where each row is a word and the
columns are as follows:

:``language``: The word's doculect.
:``iso_code``: The ISO 639-3 code of the word's doculect; can be empty.
:``gloss``: The word's meaning as described in the dataset.
:``global_id``: The Concepticon ID of the word's gloss.
:``local_id``: The dataset's ID of the word's gloss.
:``transcription``: The word's transcription in either IPA or ASJP.
:``cognate_class``: The ID of the set of cognates the word belongs to.
:``tokens``: The word's phonological segments, space-separated.
:``notes``: Field for additional information; can be empty.

The datasets are published under a `Creative Commons Attribution-ShareAlike 4.0
International License`_ and can also be found in Zenodo (URL pending).


vectors
-------

The ``data/vectors`` directory contains the samples and targets (in the machine
learning sense) derived from the datasets. Each sample represents a pair of
words from different doculects but denoting the same gloss. The sample features
are described in section 4.3 of the paper.


params
------

The ``data/params`` directory contains the parameters used for inferring the
PMI features of the aforementioned feature vectors. For more information, refer
to Jäger (2015).


code
====

The ``code`` directory contains the source code used to run the study's
experiment. It is Python 3 code and needs `LingPy`_ and `scikit-learn`_ as
direct dependencies.


setup and usage
---------------

::

    # clone this repository
    git clone https://github.com/evolaemp/svmcc
    
    # you do not need to create a virtual environment if you know what you are
    # doing; remember that the code is written in python3
    virtualenv path/to/my/venv
    source path/to/my/venv/bin/activate
    
    # install the dependencies
    pip install -r requirements.txt
    
    # use manage.py to invoke the commands
    python manage.py --help


commands
--------

TBA


licence
-------

The source code is published under the `MIT License`_.


links
=====

* Zenodo URL here

.. _`tsv`: https://en.wikipedia.org/wiki/Tab-separated_values 
.. _`Creative Commons Attribution-ShareAlike 4.0 International License`: https://creativecommons.org/licenses/by-sa/4.0/
.. _`LingPy`: https://github.com/lingpy/lingpy
.. _`scikit-learn`: https://github.com/scikit-learn/scikit-learn
.. _`MIT License`: http://choosealicense.com/licenses/mit/
