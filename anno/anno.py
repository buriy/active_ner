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
            user = project.users.all()[0]
        self.admin = user

    def get_labels(self):
        return [str(label) for label in self.project.labels.all()]

    def get_doc(self, doc, with_anno=True):
        r = []
        if with_anno:
            for ann in doc.seq_annotations.all():
                r.append([ann.start_offset, ann.end_offset, str(ann.label)])
        return {'id': doc.id, 'text': doc.text, 'labels': r, 'meta': doc.meta}

    def get_approved_doc_count(self):
        return self.project.documents.all().filter(annotations_approved_by__isnull=False).count()

    def get_unapproved_doc_count(self):
        return self.project.documents.all().filter(annotations_approved_by__isnull=True).count()

    def get_approved_docs(self, with_anno=True):
        docs = []
        for d in self.project.documents.all().filter(annotations_approved_by__isnull=False):
            docs.append(self.get_doc(d, with_anno=with_anno))
        return docs

    def get_unapproved_docs(self, with_anno=True, limit=None):
        docs = []
        items = self.project.documents.all().filter(annotations_approved_by__isnull=True)
        if limit is not None:
            items = items[:limit]
        for d in items:
            docs.append(self.get_doc(d, with_anno=with_anno))
        return docs

    def _set_annotations(self, doc, labels):
        from api.models import SequenceAnnotation
        for (start, end, label) in labels:
            sa = SequenceAnnotation(start_offset=start, end_offset=end, document=doc, user=self.admin)
            sa.label = self.label_by_name[label]
            sa.save()

    def update_doc(self, id, labels, updated=None, priority=None):
        from api.models import Document
        doc = Document.objects.get(id=id)
        if doc.annotations_approved_by_id:
            return False
        doc.seq_annotations.all().delete()
        self._set_annotations(doc, labels)
        if updated is not None:
            doc.updated_at = updated
        if priority is not None:
            doc.priority = priority
        doc.save()
        return True

    def add_doc(self, text, meta, labels, updated=None, priority=None):
        from api.models import Document
        doc = Document(text=text, meta=meta, project=self.project)
        if updated is not None:
            doc.updated_at = updated
        if priority is not None:
            doc.priority = priority
        doc.save()
        self._set_annotations(doc, labels)

    def del_unapproved(self, max_del):
        from api.models import Document
        ids = self.project.documents.all().filter(annotations_approved_by__isnull=True).values_list('id', flat=True)
        Document.objects.filter(id__in=ids[:max_del]).delete()

    def get_doc_texts(self):
        return dict(self.project.documents.all().values_list('text', 'id'))

    def fix_unapproved(self):
        from api.models import Document
        Document.objects.filter(seq_annotations__isnull=False).distinct().update(annotations_approved_by=self.admin)
