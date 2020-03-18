import os


def init_django(settings_dir="app.settings"):
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_dir)
    django.setup()


def get_project_by_name(name=None):
    from api.models import Project
    return Project.objects.get(name=name)


def get_project_by_id(id):
    from api.models import Project
    return Project.objects.get(id=id)


class Client:
    def __init__(self, project, user=None):
        init_django()
        self.project = project
        self.label_by_name = {str(label): label for label in self.project.labels.all()}
        if user is None:
            # from django.contrib.auth import get_user_model
            # User = get_user_model()
            # self.admin = User.objects.get(username='buriy')
            user = project.users.all()[0]
        self.admin = user

    def get_labels(self):
        return [str(label) for label in self.project.labels.all()]

    def get_doc(self, doc):
        r = []
        for ann in doc.seq_annotations.all():
            r.append([ann.start_offset, ann.end_offset, str(ann.label)])
        return {'text': doc.text, 'labels': r, 'meta': doc.meta}

    def get_docs(self):
        docs = []
        for d in self.project.documents.all().filter(annotations_approved_by__isnull=False):
            docs.append(self.get_doc(d))
        return docs

    def get_doc_texts(self):
        return dict(self.project.documents.all().values_list('text', 'id'))

    def add_doc(self, text, meta, labels):
        from api.models import Document, SequenceAnnotation
        doc = Document(text=text, meta=meta, project=self.project)
        doc.save()
        for (start, end, label) in labels:
            sa = SequenceAnnotation(start_offset=start, end_offset=end, document=doc, user=self.admin)
            sa.label = self.label_by_name[label]
            sa.save()

    def del_unapproved(self, max_del):
        from api.models import Document
        ids = self.project.documents.all().filter(annotations_approved_by__isnull=True).values_list('id', flat=True)
        Document.objects.filter(id__in=ids[:max_del]).delete()


    def add_docs(self, docs, max_add=None):
        existing_docs = self.get_doc_texts()
        added = 0
        for doc in docs:
            if doc['text'] in existing_docs:
                # skip the document, it was already added
                continue
            self.add_doc(doc['text'], doc.get('meta', {}), doc['labels'])
            added += 1
            if max_add and added >= max_add:
                break
        return added
