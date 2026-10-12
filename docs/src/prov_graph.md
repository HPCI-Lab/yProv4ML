
## Provenance Graph Creation (GraphViz)

The standard method to generate the .dot file containing the provenance graph is to set the `create_graph` parameter to `True`. 

If the user necessitates to turn a PROV-JSON created with yProv4ML into a .dot file, the following code command can be used: 

```bash
python -m yprov4ml.prov2dot --prov_json prov.json --output prov_graph.dot
```

## Provenance Graph Image (SVG)

The standard method to generate the .svg image of the provenance graph is to set the `create_svg` parameter to `True`.
In this case both `create_graph` and `create_svg`have to be set to `True`.

If the user necessitates to turn a .dot file into a .svg file, the following code command can be used: 

```bash
python -m yprov4ml.dot2svg --dot prov_graph.dot --output prov_graph.svg
```

Or alternatively, using directly the Graphviz suite: 

```bash
dot -Tsvg -O prov_graph.dot
```

## Workflow Card (.md)

yProv4ML additionally has a built-in script to convert run directories (or PROV-JSONL files, such as the ones created using [Flowcept](https://flowcept.org/)) to [Workflow Cards](https://arxiv.org/abs/2608.11022).

To convert a run directory, the user can run: 

```bash
python -m yprov4ml.prov2card --path **<path to jsonl or run_dir>**
```

specifying either the desired run directory (i.e. `example_0`) or a JSONL file (i.e. `flowcept_buffer.jsonl`). 


<div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
    <a href="setup.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">← Prev</a>
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">🏠 Home</a>
    <a href="logging.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">Next →</a>
</div>
