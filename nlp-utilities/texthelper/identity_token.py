
class IdentityToken(object):
    def __init__(self, id, token, start_idx, end_idx):
        self.id = id
        self.token = token
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.deleted = False

    def setId(self, id):
        self.id = id

    def getId(self):
        return self.id

    def setToken(self, token):
        self.token = token

    def getToken(self):
        return self.token

    def set_start_idx(self, start_idx):
        self.start_idx = start_idx

    def get_start_idx(self):
        return self.start_idx

    def set_end_idx(self, end_idx):
        self.end_idx = end_idx

    def get_end_idx(self):
        return self.end_idx

    def get_len(self):
        return self.end_idx - self.start_idx

    def set_deleted(self):
        self.deleted = True

    def is_deleted(self):
        return self.deleted
