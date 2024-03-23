{\rtf1\ansi\ansicpg1252\cocoartf2709
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 ArialMT;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red182\green186\blue183;\red221\green221\blue220;
\red0\green0\blue0;\red182\green186\blue183;}
{\*\expandedcolortbl;;\cssrgb\c0\c1\c1;\cssrgb\c76457\c77634\c76753\c7843;\cssrgb\c89118\c89118\c89007;
\cssrgb\c0\c0\c0;\cssrgb\c76443\c77620\c76642\c7843;}
\paperw11900\paperh16840\margl1440\margr1440\vieww24180\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
import logging\
import multiprocessing\
import os\
import sys\
from gensim.models.word2vec import LineSentence\
from gensim.models.word2vec import Word2Vec\
\
\
if __name__ == '__main__':\
\'a0 \'a0 program = os.path.basename(sys.argv[0])\
\'a0 \'a0 logger = logging.getLogger(program)\
\
\'a0 \'a0 logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')\
\'a0 \'a0 logging.root.setLevel(level=logging.INFO)\
\'a0 \'a0 logger.info("Running %s", ' '.join(sys.argv))\
\
\'a0 \'a0 # Check and process input arguments.\
\'a0 \'a0 if len(sys.argv) < 4:\
\'a0 \'a0 \'a0 \'a0 print(globals()['__doc__'] % locals())\
\'a0 \'a0 \'a0 \'a0 sys.exit(1)\
\
\'a0 \'a0 inp, outp, veco = sys.argv[1:4]\
\
\'a0 \'a0 max_length = 0\
\'a0 \'a0 with open(inp, 'r') as f:\
\'a0 \'a0 \'a0 \'a0 for line in f.readlines():\
\'a0 \'a0 \'a0 \'a0 \'a0 \'a0 max_length = max(max_length, len(line))\
\'a0 \'a0 \cf2 \cb6 logger.info\cf2 \cb3 ("Max article length: %s words.", max_length)\
\
\'a0 \'a0 params = \{\
\'a0 \'a0 \'a0 \'a0 'size': 300,\
\'a0 \'a0 \'a0 \'a0 'window': 5,\
\'a0 \'a0 \'a0 \'a0 'min_count': 1, \'a0# will not generate vector if the term does not occur at least 10 times\
\'a0 \'a0 \'a0 \'a0 'workers': max(1, multiprocessing.cpu_count() - 1),\
\'a0 \'a0 \'a0 \'a0 'sample': 1E-5,\
\'a0 \'a0 \'a0 \'a0 'sg': 1,\
\'a0 \'a0 \}\
\'a0 \'a0 word2vec = Word2Vec(LineSentence(inp, max_sentence_length=max_length), **params)\
\'a0 \'a0 word2vec.save(outp)\
\
\'a0 \'a0 if veco:\
\'a0 \'a0 \'a0 \'a0 word2vec.wv.save_word2vec_format(outp + '.model.txt', binary=False)}