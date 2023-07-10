import requests 
import pickle 
import time 
import regex as re 
from SPARQLWrapper import SPARQLWrapper, JSON
from fuzzywuzzy import fuzz

import signal
from contextlib import contextmanager

class TimedOut(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimedOut("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


object_query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://dbpedia.org/property/>

SELECT ?property ?propertyLabel ?subject ?subjectLabel
WHERE {
  <ENTITY_URI> ?property ?subject .
  
  OPTIONAL {
    ?property rdfs:label ?propertyLabel .
    FILTER (langMatches(lang(?propertyLabel), "en"))
  }
  
  OPTIONAL {
    ?subject rdfs:label ?subjectLabel .
    FILTER (langMatches(lang(?subjectLabel), "en"))
  }
  
  FILTER (
    ?property NOT IN (
      <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>,
      <http://purl.org/dc/terms/subject>, 
      <http://dbpedia.org/ontology/wikiPageWikiLink>, 
      <http://dbpedia.org/property/wikiPageUsesTemplate>,
      <http://dbpedia.org/ontology/wikiPageRedirects>,
      <http://dbpedia.org/property/align>,
      <http://dbpedia.org/property/caption>,
      <http://dbpedia.org/property/format>,
      <http://dbpedia.org/property/float>,
      <http://dbpedia.org/property/footer>,
      <http://dbpedia.org/property/image>,
      <http://dbpedia.org/property/width>,
      <http://dbpedia.org/property/totalWidth>,
      <http://dbpedia.org/property/imageCaption>,
      <http://dbpedia.org/property/filename>,
      <http://dbpedia.org/property/singleLine>,
      <http://dbpedia.org/ontology/wikiPageDisambiguates>
    )   &&
    REGEX(STR(?property), "^http://dbpedia.org/")
  )
}
"""


subject_query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://dbpedia.org/property/>

SELECT ?property ?propertyLabel ?subject ?subjectLabel
WHERE {
  ?subject ?property <ENTITY_URI> .
  
  OPTIONAL {
    ?property rdfs:label ?propertyLabel .
    FILTER (langMatches(lang(?propertyLabel), "en"))
  }
  
  OPTIONAL {
    ?subject rdfs:label ?subjectLabel .
    FILTER (langMatches(lang(?subjectLabel), "en"))
  }
  
  FILTER (
    ?property NOT IN (
      <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>,
      <http://purl.org/dc/terms/subject>, 
      <http://dbpedia.org/ontology/wikiPageWikiLink>, 
      <http://dbpedia.org/property/wikiPageUsesTemplate>,
      <http://dbpedia.org/ontology/wikiPageRedirects>,
      <http://dbpedia.org/property/align>,
      <http://dbpedia.org/property/caption>,
      <http://dbpedia.org/property/format>,
      <http://dbpedia.org/property/float>,
      <http://dbpedia.org/property/footer>,
      <http://dbpedia.org/property/image>,
      <http://dbpedia.org/property/width>,
      <http://dbpedia.org/property/totalWidth>,
      <http://dbpedia.org/property/imageCaption>,
      <http://dbpedia.org/property/filename>,
      <http://dbpedia.org/property/singleLine>,
      <http://dbpedia.org/ontology/wikiPageDisambiguates>
    )   &&
    REGEX(STR(?property), "^http://dbpedia.org/")
  )
}
"""

sparql = SPARQLWrapper("https://dbpedia.org/sparql")
all_content_uris = pickle.load(open('all_content_uris_sub.pkl', 'rb'))

not_found_uris = {'en': {}, 'pt': {}, 'de': {}}

for lang in all_content_uris:
    print(lang)
    for i, entity in enumerate(all_content_uris[lang]):
        print(i, entity)
        time.sleep(3)
        try:
            with time_limit(3):
                query = re.sub("ENTITY_URI", all_content_uris[lang][entity]['uri'], subject_query)
                sparql.setQuery(query)
                sparql.setReturnFormat(JSON)
                results = sparql.query().convert()
                result_list_object = [] 
                for result in results['results']['bindings']:
                    if('propertyLabel' in result and 'subjectLabel' in result):
                        result_list_object.append((result['propertyLabel']['value'], result['subjectLabel']['value']))
                all_content_uris[lang][entity]['object_properties'] = result_list_object
        except TimedOut as e:   
            print("Timed out object!")
            not_found_uris[lang][entity] = all_content_uris[lang][entity]

        time.sleep(2)   

dump = open('all_content_uris_sub.pkl', 'wb')
pickle.dump(all_content_uris, dump)
dump.close()

dump_not_found_subject = open('not_found_uris_subject.pkl', 'wb')
pickle.dump(not_found_uris, dump_not_found_subject)
dump_not_found_subject.close()
