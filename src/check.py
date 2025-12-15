from clearml import Model

ms = Model.query_models(project_name="Lab3", only_published=True)
print("Found", len(ms), "production models")
for m in ms:
    print(m.name, m.id, m.url)