
import os
import prov
import prov.model as prov
from datetime import datetime
from mlflow.entities import Run

from ..constants import PROV4ML_DATA
from ..datamodel.lod import Prov4MLLOD
from ..datamodel.artifact_data import artifact_is_pytorch_model
from ..provenance.context import Context

def first_level_prov(
        run:Run, 
        doc: prov.ProvDocument,
    ) -> prov.ProvDocument:
    """
    Generates the first level of provenance for a given run.

    Args:
        run (Run): The run object.
        doc (prov.ProvDocument): The provenance document.

    Returns:
        prov.ProvDocument: The provenance document.
    """

    run_entity = doc.entity(f'{run.info.run_name}',other_attributes={
        "mlflow:run_id": Prov4MLLOD.get_lv1_attr(run.info.run_id),
        "mlflow:artifact_uri":Prov4MLLOD.get_lv1_attr(run.info.artifact_uri),
        "prov-ml:type": Prov4MLLOD.get_lv1_attr("LearningStage"),
        "mlflow:user_id": Prov4MLLOD.get_lv1_attr(run.info.user_id),
        "prov:level": Prov4MLLOD.LVL_1,
    })
        
    global_rank = os.getenv("SLURM_PROCID", None)
    if global_rank:
        node_rank = os.getenv("SLURM_NODEID", None)
        local_rank = os.getenv("SLURM_LOCALID", None) 
        run_entity.add_attributes({
            "mlflow:global_rank":Prov4MLLOD.get_lv1_attr(global_rank),
            "mlflow:local_rank":Prov4MLLOD.get_lv1_attr(local_rank),
            "mlflow:node_rank":Prov4MLLOD.get_lv1_attr(node_rank),
        })

    run_activity = doc.activity(f'{run.info.run_name}_execution', other_attributes={
        'prov-ml:type': Prov4MLLOD.get_lv1_attr("LearningExecution"),
        "prov:level": Prov4MLLOD.LVL_1
    })
    #experiment entity generation
    experiment = doc.entity(PROV4ML_DATA.experiment_name,other_attributes={
        "prov-ml:type": Prov4MLLOD.get_lv1_attr("Experiment"),
        "mlflow:experiment_id": Prov4MLLOD.get_lv1_attr(run.info.experiment_id),
        "prov:level":Prov4MLLOD.LVL_1
    })

    user_ag = doc.agent(f'{run.info.user_id}',other_attributes={
        "prov:level":Prov4MLLOD.LVL_1,
    })
    doc.wasAssociatedWith(f'{run.info.run_name}_execution',user_ag,other_attributes={
        "prov:level":Prov4MLLOD.LVL_1,
    })
    doc.entity('source_code',{
        "mlflow:source_name": Prov4MLLOD.get_lv1_attr(run.data.tags['mlflow.source.name']),
        "mlflow:source_type": Prov4MLLOD.get_lv1_attr(run.data.tags['mlflow.source.type']),
        'prov:level':Prov4MLLOD.LVL_1,   
    })

    if 'mlflow.source.git.commit' in run.data.tags.keys():
        doc.activity('commit',other_attributes={
            "mlflow:source_git_commit": Prov4MLLOD.get_lv2_attr(run.data.tags['mlflow.source.git.commit']),
            'prov:level':Prov4MLLOD.LVL_1,
        })
        doc.wasGeneratedBy('source_code','commit',other_attributes={'prov:level':Prov4MLLOD.LVL_1})
        doc.wasInformedBy(run_activity,'commit',other_attributes={'prov:level':Prov4MLLOD.LVL_1})
    else:
        doc.used(run_activity,'source_code',other_attributes={'prov:level':Prov4MLLOD.LVL_1})


    doc.hadMember(experiment,run_entity).add_attributes({'prov:level':Prov4MLLOD.LVL_1})
    doc.wasGeneratedBy(run_entity,run_activity,other_attributes={'prov:level':Prov4MLLOD.LVL_1})

    for name, metric in PROV4ML_DATA.metrics.items():
        for epoch, metric_per_epoch in metric.epochDataList.items():
            if metric.context == Context.TRAINING: 
                if not doc.get_record(f'epoch_{epoch}'):
                    train_activity=doc.activity(f'epoch_{epoch}',other_attributes={
                    "prov-ml:type": Prov4MLLOD.get_lv1_attr("TrainingExecution"),
                    'prov:level':Prov4MLLOD.LVL_1,
                    })
                    doc.wasStartedBy(train_activity,run_activity,other_attributes={'prov:level':Prov4MLLOD.LVL_1})

                metric_entity = doc.entity(f'{name}',{
                    'prov-ml:type':Prov4MLLOD.get_lv1_attr('Metric'),
                    'prov:level':Prov4MLLOD.LVL_1,
                })

                for metric_value in metric_per_epoch:
                    metric_entity.add_attributes({
                        'mlflow:step-value':Prov4MLLOD.get_lv1_epoch_value(epoch, metric_value),
                    })
                doc.wasGeneratedBy(metric_entity,f'epoch_{epoch}',
                                    identifier=f'{name}_{epoch}_gen',
                                    other_attributes={'prov:level':Prov4MLLOD.LVL_1})
                
            # elif metric.context == Context.VALIDATION:
            #     if not doc.get_record(f'epoch_{epoch}'):
            #         train_activity=doc.activity(f'epoch_{epoch}',other_attributes={
            #         "prov-ml:type": Prov4MLLOD.get_lv1_attr("ValidationExecution"),
            #         'prov:level':Prov4MLLOD.LVL_1,
            #         })
            #         doc.wasStartedBy(train_activity,run_activity,other_attributes={'prov:level':Prov4MLLOD.LVL_1})

            #     metric_entity = doc.entity(f'{name}',{
            #         'prov-ml:type':Prov4MLLOD.get_lv1_attr('Metric'),
            #         'prov:level':Prov4MLLOD.LVL_1,
            #     })

            #     for metric_value in metric_per_epoch:
            #         metric_entity.add_attributes({
            #             'mlflow:step-value':Prov4MLLOD.get_lv1_epoch_value(epoch, metric_value),
            #         })
            #     doc.wasGeneratedBy(metric_entity,f'epoch_{epoch}',
            #                         identifier=f'{name}_{epoch}_gen',
            #                         other_attributes={'prov:level':Prov4MLLOD.LVL_1})
                
            elif metric.context == Context.EVALUATION:
                if not doc.get_record(f'eval'):
                    eval_activity=doc.activity(f'eval',other_attributes={
                    "prov-ml:type": Prov4MLLOD.get_lv1_attr("EvaluationExecution"),
                    'prov:level':Prov4MLLOD.LVL_1,
                    })
                    doc.wasStartedBy(eval_activity,run_activity,other_attributes={'prov:level':Prov4MLLOD.LVL_1})

                metric_entity = doc.entity(f'{name}',{
                    'prov-ml:type':Prov4MLLOD.get_lv1_attr('Metric'),
                    'prov:level':Prov4MLLOD.LVL_1,
                })

                for metric_value in metric_per_epoch:
                    metric_entity.add_attributes({
                        'mlflow:step-value':Prov4MLLOD.get_lv1_epoch_value(epoch, metric_value),
                    })
                doc.wasGeneratedBy(metric_entity,f'eval',
                                    identifier=f'eval_gen',
                                    other_attributes={'prov:level':Prov4MLLOD.LVL_1})


                        
    for name, param in PROV4ML_DATA.parameters.items():
        ent = doc.entity(f'{name}',{
            'mlflow:value': Prov4MLLOD.get_lv1_attr(param.value),
            'prov-ml:type': Prov4MLLOD.get_lv1_attr('Parameter'),
            'prov:level':Prov4MLLOD.LVL_1,
        })
        doc.used(run_activity,ent,other_attributes={'prov:level':Prov4MLLOD.LVL_1})

    #dataset entities generation
    ent_ds = doc.entity(f'dataset',other_attributes={'prov:level':Prov4MLLOD.LVL_1})
    for dataset_input in run.inputs.dataset_inputs:
        attributes={
            'prov-ml:type': Prov4MLLOD.get_lv1_attr('Dataset'),
            'mlflow:digest': Prov4MLLOD.get_lv1_attr(dataset_input.dataset.digest),
            'prov:level':Prov4MLLOD.LVL_1,
        }

        ent= doc.entity(f'{dataset_input.dataset.name}-{dataset_input.dataset.digest}',attributes)
        doc.used(run_activity,ent, other_attributes={'prov:level':Prov4MLLOD.LVL_1})
        doc.wasDerivedFrom(ent,ent_ds,identifier=f'{dataset_input.dataset.name}-{dataset_input.dataset.digest}_der',other_attributes={'prov:level':Prov4MLLOD.LVL_1})
    

    #model version entities generation
    model_version = PROV4ML_DATA.get_final_model()
    model_entity_label = model_version.path
    modv_ent=doc.entity(model_entity_label,{
        "prov-ml:type": Prov4MLLOD.get_lv1_attr("ModelVersion"),
        'prov-ml:creation_epoch': Prov4MLLOD.get_lv1_attr(model_version.step),
        'prov-ml:artifact_uri': Prov4MLLOD.get_lv1_attr(model_version.path),
        'prov-ml:creation_timestamp': Prov4MLLOD.get_lv1_attr(datetime.fromtimestamp(model_version.creation_timestamp / 1000)),
        'prov-ml:last_modified_timestamp': Prov4MLLOD.get_lv1_attr(datetime.fromtimestamp(model_version.last_modified_timestamp / 1000)),
        'prov:level': Prov4MLLOD.LVL_1
    })
    doc.wasGeneratedBy(modv_ent,run_activity,identifier=f'{model_entity_label}_gen',other_attributes={'prov:level':Prov4MLLOD.LVL_1})
    
    model_ser = doc.activity(f'mlflow:ModelRegistration',other_attributes={'prov:level':Prov4MLLOD.LVL_1})
    doc.wasInformedBy(model_ser,run_activity,other_attributes={'prov:level':Prov4MLLOD.LVL_1})
    doc.wasGeneratedBy(model_entity_label,model_ser,other_attributes={'prov:level':Prov4MLLOD.LVL_1})
    
    for artifact in PROV4ML_DATA.get_model_versions()[:-1]: 
        memb=doc.hadMember(model_entity_label,f"{artifact.path}")
        memb.add_attributes({'prov:level':Prov4MLLOD.LVL_1})
    

    doc.activity("data_preparation",other_attributes={
        "prov-ml:type":"FeatureExtractionExecution",
        'prov:level':Prov4MLLOD.LVL_1,
    })
    #add attributes to dataset entities
    for dataset_input in run.inputs.dataset_inputs:
        attributes={
            'mlflow:profile': Prov4MLLOD.get_lv2_attr(dataset_input.dataset.profile),
            'mlflow:schema': Prov4MLLOD.get_lv2_attr(dataset_input.dataset.schema),
        }
        ent= doc.get_record(f'{dataset_input.dataset.name}-{dataset_input.dataset.digest}')[0]
        ent.add_attributes(attributes)

        doc.wasGeneratedBy(ent,'data_preparation',other_attributes={'prov:level':Prov4MLLOD.LVL_1})        #use two binary relation for yProv
    doc.used('data_preparation','dataset',other_attributes={'prov:level':Prov4MLLOD.LVL_1})
    
    #artifact entities generation
    for artifact in PROV4ML_DATA.get_artifacts():
        ent=doc.entity(f'{artifact.path}',{
            'mlflow:artifact_path': Prov4MLLOD.get_lv1_attr(artifact.path),
            'prov:level':Prov4MLLOD.LVL_1,
            #the FileInfo object stores only size and path of the artifact, specific connectors to the artifact store are needed to get other metadata
        })
        if artifact_is_pytorch_model(artifact):
            doc.wasGeneratedBy(f"{artifact.path}", model_ser,other_attributes={'prov:level':Prov4MLLOD.LVL_1})
        else: 
            doc.wasGeneratedBy(ent,run_activity,identifier=f'{artifact.path}_gen',other_attributes={'prov:level':Prov4MLLOD.LVL_1})    

    return doc
