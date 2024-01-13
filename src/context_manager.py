from contextlib import contextmanager
import mlflow
import prov.model as prov
import prov.dot as dot
from datetime import datetime
import ast

def traverse_artifact_tree(client:mlflow.MlflowClient,run_id:str,path=None) -> [mlflow.entities.file_info.FileInfo]:
    #Traversal of the artifact tree of a run, stored as an acyclic graph
    artifact_list=client.list_artifacts(run_id,path)
    artifact_paths=[]
    for artifact in artifact_list:
        if artifact.is_dir:
            artifact_paths.extend(traverse_artifact_tree(client,run_id,artifact.path))
        else:
            artifact_paths.append(artifact)
    return artifact_paths

@contextmanager
def start_run(id=None,run_name=None):

    act_run= mlflow.start_run(id,run_name=run_name) #start the run
    print('started run', act_run.info.run_id)
    yield act_run #return the mlflow context manager, same one as mlflow.start_run()


    run_id=act_run.info.run_id
    
    mlflow.end_run() #end the run, as per mlflow documentation
    print('ended run')

    print('doc generation')
    client = mlflow.MlflowClient()
    act_run=client.get_run(run_id)
    doc = prov.ProvDocument()

    #set namespaces
    doc.add_namespace('prov','http://www.w3.org/ns/prov#')
    doc.add_namespace('xsd','http://www.w3.org/2000/10/XMLSchema#')
    
    doc.add_namespace('mlflow', ' ') #TODO: find namespaces of mlflow and prov-ml ontologies
    doc.add_namespace('prov-ml', 'p')

    doc.add_namespace('ex','http://www.example.org')


    run_activity = doc.activity(f'ex:{act_run.info.run_name}',
                                datetime.fromtimestamp(act_run.info.start_time/1000),datetime.fromtimestamp(act_run.info.end_time/1000),
                                other_attributes={
        'prov-ml:type':'LearningStageExecution',
        "mlflow:experiment_id":str(act_run.info.experiment_id),
        "mlflow:run_id": str(act_run.info.run_id),
        "mlflow:artifact_uri":str(act_run.info.artifact_uri)
    })

    #get dataset data
    ent_ds = doc.entity(f'ex:dataset')
    for dataset_input in act_run.inputs.dataset_inputs:

        #dataset tracking is still experimental, tags are stored in a serialized dict
        tags=ast.literal_eval(dataset_input.dataset.source)
        #source_commit=tags['tags']['mlflow.source.git.commit']
        attributes={
            'prov-ml:type':'FeatureSetData',
            'mlflow:digest':str(dataset_input.dataset.digest),
            #'mlflow:source_commit':source_commit,   
        }

        #datasets are associated with two sets of tags: input tags, of the DatasetInput object, and the tags of the dataset itself
        for input_tag in dataset_input.tags:
            attributes[f'mlflow:{input_tag.key.strip("mlflow.")}']=str(input_tag.value)
        for key,value in tags['tags'].items():
            attributes[f'mlflow:{str(key).strip("mlflow.")}']=str(value)
        
        ent= doc.entity(f'mlflow:{dataset_input.dataset.name}-{dataset_input.dataset.digest}',attributes)
        doc.used(run_activity,ent)
        doc.wasDerivedFrom(ent,ent_ds)
    
    for name,value in act_run.data.params.items():
        ent = doc.entity(f'ex:{name}',{
            'ex:value':value,
            'prov-ml:type':'LearningHyperparameterValue'
        })
        doc.used(run_activity,ent)
    
    for name,value in act_run.data.metrics.items():
        #the Run object stores only the most recent metrics, to get all metrics lower level API is needed
        for metric in client.get_metric_history(act_run.info.run_id,name):
            ent=doc.entity(f'ex:{name}_{metric.step}',{
                'ex:value':metric.value,
                'ex:epoch':metric.step,
                'prov-ml:type':'ModelEvaluation'
            })
            doc.wasGeneratedBy(ent,run_activity,datetime.fromtimestamp(metric.timestamp/1000))

    model_version = client.search_model_versions(f'run_id="{run_id}"')[0] #only one model version per run (in this case)
    mod_ser=doc.activity('mlflow:ModelSerialization')
    modv_ent=doc.entity(f'ex:{model_version.name}_{model_version.version}',{
        'mlflow:version':str(model_version.version),
        'mlflow:artifact_uri':str(model_version.source),
        'mlflow:creation_timestamp':str(datetime.fromtimestamp(model_version.creation_timestamp/1000)),
        'mlflow:last_updated_timestamp':str(datetime.fromtimestamp(model_version.last_updated_timestamp/1000)),
    })
    
    #get the model registered in the model registry of mlflow
    model = client.get_registered_model(model_version.name)
    mod_ent=doc.entity(f'ex:{model.name}',{
        'mlflow:creation_timestamp':str(datetime.fromtimestamp(model.creation_timestamp/1000))
    })
    
    doc.wasGeneratedBy(modv_ent,mod_ser)
    doc.wasStartedBy(mod_ser,run_activity)
    doc.specializationOf(modv_ent,mod_ent)

    #artifacts are stored in a directory tree, this function traverses the tree and returns a list of artifacts
    artifacts=traverse_artifact_tree(client,run_id)
    for artifact in artifacts:
        ent=doc.entity(f'ex:{artifact.path}',{
            'mlflow:artifact_path':artifact.path,
            #the FileInfo object stores only size and path of the artifact, specific connectors to the artifact store are needed to get other metadata
        })
        doc.wasGeneratedBy(ent,run_activity)

    with open('prov_graph.json','w') as prov_graph:
        doc.serialize(prov_graph)
    with open('prov_graph.dot', 'w') as prov_graph:
        prov_graph.write(dot.prov_to_dot(doc).to_string())


