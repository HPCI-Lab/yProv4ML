
## Provenance Graph Creation (GraphViz)

The standard method to generate the .dot file containing the provenance graph is to set the `create_graph` parameter to `True`. 

If the user necessitates to turn a PROV-JSON created with yProv4ML into a .dot file, the following code command can be used: 

```bash
python -m prov4ml.prov2dot --prov_json prov.json --output prov_graph.dot
```

## Provenance Graph Image (SVG)

The standard method to generate the .svg image of the provenance graph is to set the `create_svg` parameter to `True`.
In this case both `create_graph` and `create_svg`have to be set to `True`.

If the user necessitates to turn a .dot file into a .svg file, the following code command can be used: 

```bash
python -m prov4ml.dot2svg --dot prov_graph.dot --output prov_graph.svg
```

Or alternatively, using directly the Graphviz suite: 

```bash
dot -Tsvg -O prov_graph.dot
```

[Home](README.md) | [Prev](setup.md) | [Next](logging.md)
