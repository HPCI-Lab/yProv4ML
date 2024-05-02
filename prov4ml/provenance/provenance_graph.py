
import warnings
import mlflow
import prov
import prov.model as prov
from datetime import datetime
from mlflow.entities import Run
from mlflow.entities.file_info import FileInfo
from collections import namedtuple
from typing import List, Optional

from ..logging import Context

lv_attr = namedtuple('lv_attr', ['level', 'value'])
LVL_1 = "1"
LVL_2 = "2"


def traverse_artifact_tree(
        client:mlflow.MlflowClient,
        run_id:str,path=None
        ) -> List[FileInfo]:
    """
    Recursively traverses the artifact tree of a given run in MLflow and returns a list of FileInfo objects.

    Args:
        client (mlflow.MlflowClient): The MLflow client object.
        run_id (str): The ID of the run.
        path (str, optional): The path to start the traversal from. Defaults to None.

    Returns:
        List[FileInfo]: A list of FileInfo objects representing the artifacts in the tree.
    """    
    artifact_list=client.list_artifacts(run_id,path)
    artifact_paths=[]
    for artifact in artifact_list:
        if artifact.is_dir:
            artifact_paths.extend(traverse_artifact_tree(client,run_id,artifact.path))
        else:
            artifact_paths.append(artifact)
    return artifact_paths

def first_level_prov(
        run:Run, 
        doc: prov.ProvDocument,
        global_rank: Optional[str] = None,
        local_rank: Optional[str] = None,
        node_rank: Optional[str] = None,
    ) -> prov.ProvDocument:
    """
    Generates the first level of provenance for a given run.

    Args:
        run (Run): The run object.
        doc (prov.ProvDocument): The provenance document.

    Returns:
        prov.ProvDocument: The provenance document.
    """
    client = mlflow.MlflowClient()

    if global_rank is None:
        run_entity = doc.entity(f'{run.info.run_name}',other_attributes={
            "mlflow:run_id": str(lv_attr(LVL_1,str(run.info.run_id))),
            "mlflow:artifact_uri":str(lv_attr(LVL_1,str(run.info.artifact_uri))),
            "prov-ml:type":str(lv_attr(LVL_1,"LearningStage")),
            "mlflow:user_id":str(lv_attr(LVL_1,str(run.info.user_id))),
            "prov:level":LVL_1
        })
    else:
        run_entity = doc.entity(f'{run.info.run_name}',other_attributes={
            "mlflow:run_id": str(lv_attr(LVL_1,str(run.info.run_id))),
            "mlflow:artifact_uri":str(lv_attr(LVL_1,str(run.info.artifact_uri))),
            "prov-ml:type":str(lv_attr(LVL_1,"LearningStage")),
            "mlflow:user_id":str(lv_attr(LVL_1,str(run.info.user_id))),
            "prov:level":LVL_1, 
            "mlflow:global_rank":str(lv_attr(LVL_1,global_rank)),
            "mlflow:local_rank":str(lv_attr(LVL_1,local_rank)),
            "mlflow:node_rank":str(lv_attr(LVL_1,node_rank)),
        })

    run_activity = doc.activity(f'{run.info.run_name}_execution',
                                #datetime.fromtimestamp(run.info.start_time/1000),
                                #datetime.fromtimestamp(run.info.end_time/1000),
                                other_attributes={
        'prov-ml:type':str(lv_attr(LVL_1,'LearningStageExecution')),
        "prov:level":LVL_1
    })
    #experiment entity generation
    experiment = doc.entity(f'{client.get_experiment(run.info.experiment_id).name}',other_attributes={
        "prov-ml:type":str(lv_attr(LVL_1,"LearningExperiment")),
        "mlflow:experiment_id": str(lv_attr(LVL_1,str(run.info.experiment_id))),
        "prov:level":LVL_1
    })

    doc.hadMember(experiment,run_entity).add_attributes({
        'prov:level':LVL_1
    })
    doc.wasGeneratedBy(run_entity,run_activity,other_attributes={
        'prov:level':LVL_1
    })


    #metrics and params generation
    for name,_ in run.data.metrics.items():
        #the Run object stores only the most recent metrics, to get all metrics lower level API is needed
        for metric in client.get_metric_history(run.info.run_id,name):
            i=0
            ent=doc.entity(f'{name}_{metric.step or i}',{
                'prov-ml:type':'ModelEvaluation',
                'mlflow:value':str(lv_attr(LVL_1,metric.value)),
                'mlflow:step':str(lv_attr(LVL_1,metric.step or i)),
                'prov:level':LVL_1,
            })
            doc.wasGeneratedBy(ent,run_activity,
                               #datetime.fromtimestamp(metric.timestamp/1000),
                               identifier=f'{name}_{metric.step}_gen',
                               other_attributes={
                                    'prov:level':LVL_1
                               })
            i+=1

    for name,value in run.data.params.items():
        ent = doc.entity(f'{name}',{
            'mlflow:value':str(lv_attr(LVL_1,value)),
            'prov-ml:type':str(lv_attr(LVL_1,'LearningHyperparameterValue')),
            'prov:level':LVL_1,
        })
        doc.used(run_activity,ent,other_attributes={'prov:level':LVL_1})

    #dataset entities generation
    ent_ds = doc.entity(f'dataset',other_attributes={'prov:level':LVL_1})
    for dataset_input in run.inputs.dataset_inputs:
        attributes={
            'prov-ml:type':str(lv_attr(LVL_1,'FeatureSetData')),
            'mlflow:digest':str(lv_attr(LVL_1,str(dataset_input.dataset.digest))),
            'prov:level':LVL_1,
        }

        ent= doc.entity(f'{dataset_input.dataset.name}-{dataset_input.dataset.digest}',attributes)
        doc.used(run_activity,ent, other_attributes={'prov:level':LVL_1})
        doc.wasDerivedFrom(ent,ent_ds,identifier=f'{dataset_input.dataset.name}-{dataset_input.dataset.digest}_der',other_attributes={'prov:level':LVL_1})
    

    #model version entities generation
    models_saved = client.search_model_versions(f'run_id="{run.info.run_id}"')
    if len(models_saved) > 0:
        model_version = models_saved[0] #only one model version per run (in this case)
    
        modv_ent=doc.entity(f'{model_version.name}_{model_version.version}',{
            "prov-ml:type":str(lv_attr(LVL_1,"Model")),
            'mlflow:version':str(lv_attr(LVL_1,model_version.version)),
            'mlflow:artifact_uri':str(lv_attr(LVL_1,model_version.source)),
            'mlflow:creation_timestamp':str(lv_attr(LVL_1,datetime.fromtimestamp(model_version.creation_timestamp/1000))),
            'mlflow:last_updated_timestamp':str(lv_attr(LVL_1,datetime.fromtimestamp(model_version.last_updated_timestamp/1000))),
            'prov:level':LVL_1
        })
        doc.wasGeneratedBy(modv_ent,run_activity,identifier=f'{model_version.name}_{model_version.version}_gen',other_attributes={'prov:level':LVL_1})
        
        model = client.get_registered_model(model_version.name)
        mod_ent=doc.entity(f'{model.name}',{
            "prov-ml:type":str(lv_attr(LVL_1,"Model")),
            'mlflow:creation_timestamp':str(lv_attr(LVL_1,datetime.fromtimestamp(model.creation_timestamp/1000))),
            'prov:level':LVL_1,
        })
        spec=doc.specializationOf(modv_ent,mod_ent)
        spec.add_attributes({'prov:level':LVL_1})   #specilizationOf doesn't accept other_attributes, but its cast as record does
    else:
        warnings.warn(f"No model version found for run {run.info.run_id}. Did you remember to call prov4ml.log_model()?")

    #artifact entities generation
    artifacts=traverse_artifact_tree(client,run.info.run_id)
    for artifact in artifacts:
        ent=doc.entity(f'{artifact.path}',{
            'mlflow:artifact_path':str(lv_attr(LVL_1,artifact.path)),
            'prov:level':LVL_1,
            #the FileInfo object stores only size and path of the artifact, specific connectors to the artifact store are needed to get other metadata
        })
        doc.wasGeneratedBy(ent,run_activity,identifier=f'{artifact.path}_gen',other_attributes={'prov:level':LVL_1})
    

    return doc

def second_level_prov(run:Run, doc: prov.ProvDocument) -> prov.ProvDocument:
    """
    Generates the second level of provenance for a given run.
    Args:
        run (Run): The run object.
        doc (prov.ProvDocument): The provenance document.
    Returns:
        prov.ProvDocument: The provenance document.
    """
    client = mlflow.MlflowClient()
        
    run_activity= doc.get_record(f'{run.info.run_name}_execution')[0]
    run_activity.add_attributes({
        "mlflow:status":str(lv_attr(LVL_2,run.info.status)),
        "mlflow:lifecycle_stage":str(lv_attr(LVL_2,run.info.lifecycle_stage)),
    })
    user_ag = doc.agent(f'{run.info.user_id}',other_attributes={
        "prov:level":LVL_2,
    })
    doc.wasAssociatedWith(f'{run.info.run_name}_execution',user_ag,other_attributes={
        "prov:level":LVL_2,
    })

    doc.entity('source_code',{
        "mlflow:source_name":str(lv_attr(LVL_2,run.data.tags['mlflow.source.name'])),
        "mlflow:source_type":str(lv_attr(LVL_2,run.data.tags['mlflow.source.type'])),  
        'prov:level':LVL_2,   
    })

    if 'mlflow.source.git.commit' in run.data.tags.keys():
        doc.activity('commit',other_attributes={
            "mlflow:source_git_commit":str(lv_attr(LVL_2,run.data.tags['mlflow.source.git.commit'])),
            'prov:level':LVL_2,
        })
        doc.wasGeneratedBy('source_code','commit',other_attributes={'prov:level':LVL_2})
        doc.wasInformedBy(run_activity,'commit',other_attributes={'prov:level':LVL_2})
    else:
        doc.used(run_activity,'source_code',other_attributes={'prov:level':LVL_2})

    #remove relations between metrics and run


    #create activities for training and evaluation and associate metrics

    for name,_ in run.data.metrics.items():
        for metric in client.get_metric_history(run.info.run_id,name):
            if not doc.get_record(f'train_step_{metric.step}'):
                train_activity=doc.activity(f'train_step_{metric.step}',other_attributes={
                "prov-ml:type":str(lv_attr(LVL_2,"TrainingExecution")),
                'prov:level':LVL_2,
                })
                test_activity=doc.activity(f'test_step_{metric.step}',other_attributes={
                    "prov-ml:type":str(lv_attr(LVL_2,"EvaluationExecution")),
                    'prov:level':LVL_2,
                })
                doc.wasStartedBy(train_activity,run_activity,other_attributes={'prov:level':LVL_2})
                doc.wasStartedBy(test_activity,run_activity,other_attributes={'prov:level':LVL_2})

            # if doc.get_record(f'{name}_{metric.step}_gen')[0]:
            #     doc._records.remove(doc.get_record(f'{name}_{metric.step}_gen')[0]) #accessing private attribute, propriety doesn't allow to remove records, but we need to remove the lv1 generation
            if run.data.tags[f'metric.context.{metric.key}']==Context.TRAINING.name:
                doc.wasGeneratedBy(f'{metric.key}_{metric.step}',f'train_step_{metric.step}',other_attributes={'prov:level':LVL_2})    
            elif run.data.tags[f'metric.context.{metric.key}']==Context.EVALUATION.name:
                doc.wasGeneratedBy(f'{metric.key}_{metric.step}',f'test_step_{metric.step}',other_attributes={'prov:level':LVL_2})
    
    #data transformation activity
    doc.activity("data_preparation",other_attributes={
        "prov-ml:type":"FeatureExtractionExecution",
        'prov:level':LVL_2,
    })
    #add attributes to dataset entities
    for dataset_input in run.inputs.dataset_inputs:
        attributes={
            'mlflow:profile':str(lv_attr(LVL_2,dataset_input.dataset.profile)),
            'mlflow:schema':str(lv_attr(LVL_2,dataset_input.dataset.schema)),   
        }
        ent= doc.get_record(f'{dataset_input.dataset.name}-{dataset_input.dataset.digest}')[0]
        ent.add_attributes(attributes)

        #remove old generation relationship
        # if doc.get_record(f'{dataset_input.dataset.name}-{dataset_input.dataset.digest}_der')[0]:
        #     doc._records.remove(doc.get_record(f'{dataset_input.dataset.name}-{dataset_input.dataset.digest}_der')[0])
        #doc.wasDerivedFrom(ent,'dataset','data_preparation',other_attributes={'prov:level':LVL_2})  #use new transform activity for derivation
        doc.wasGeneratedBy(ent,'data_preparation',other_attributes={'prov:level':LVL_2})        #use two binary relation for yProv
    doc.used('data_preparation','dataset',other_attributes={'prov:level':LVL_2})
    # doc.get_record('dataset')[0].add_attributes({
    #     'source_mirror':str(run.inputs.dataset_inputs[0].tags[1]),
    # })
    
    model_versions = client.search_model_versions(f'run_id="{run.info.run_id}"')
    if len(model_versions) > 0:
        model_version = model_versions[0]
        # if doc.get_record(f'{model_version.name}_{model_version.version}_gen')[0]:
        #     doc._records.remove(doc.get_record(f'{model_version.name}_{model_version.version}_gen')[0])

        model_ser = doc.activity(f'mlflow:ModelRegistration',other_attributes={'prov:level':LVL_2})
        doc.wasInformedBy(model_ser,run_activity,other_attributes={'prov:level':LVL_2})
        doc.wasGeneratedBy(f'{model_version.name}_{model_version.version}',model_ser,other_attributes={'prov:level':LVL_2})
        
        for artifact in traverse_artifact_tree(client,run.info.run_id,model_version.name): #get artifacts whose path starts with TinyVGG: these are model serialization and metadata files
            # if doc.get_record(f'{artifact.path}_gen'):
            #     doc._records.remove(doc.get_record(f'{artifact.path}_gen')[0])
            memb=doc.hadMember(f'{model_version.name}_{model_version.version}',f"{artifact.path}")
            memb.add_attributes({'prov:level':LVL_2})
    else: 
        warnings.warn(f"No model version found for run {run.info.run_id}. Did you remember to call prov4ml.log_model()?")
    return doc
