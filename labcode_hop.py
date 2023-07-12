import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from DataHandler import negSamp, transpose, DataHandler, transToLsts, DataHandler_ciaoAndEpinions
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
from multiprocessing import Process, Manager, Array
import scipy.spatial
# import tf_agent

def count_flops(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPs: {}'.format(flops.total_float_ops))



class Recommender:
    def __init__(self, sess, handler):
        self.sess = sess  # session
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', len(self.handler.trnMat.data))
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        early_stop_loss_min = 0
        wait_num = 0
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            # stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
            stloc = 0
        else:
            stloc = 0  # start location
            init = tf.global_variables_initializer()
            self.sess.run(init)
            log('Variables Inited')
        for ep in range(stloc, args.epoch):

            test = (ep % args.tstEpoch == 0)  # bool flag:test at current epoch
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, test))

            # --------- early stop ---------------

            if early_stop_loss_min == 0:
                early_stop_loss_min = reses['Loss']
            if reses['Loss'] < early_stop_loss_min:
                early_stop_loss_min = reses['Loss']
                wait_num = 0
                if ep > args.save_start:
                    self.saveHistory()
                    reses = self.testEpoch()
                    log(self.makePrint('Test', ep, reses, True))
            else:
                wait_num = wait_num + 1
                if wait_num > args.patience:
                    print(f"early stopping at epoch {ep}")
                    args.load_model = args.save_path
                    self.loadModel()
                    reses = self.testEpoch()
                    log(self.makePrint('Test', args.epoch, reses, True))
                    return

            # ------------------------------
            # if test:
            #     reses = self.testEpoch()
            #     log(self.makePrint('Test', ep, reses, test))
            # if ep % args.tstEpoch == 0:
            #     self.saveHistory()
            print()
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        self.saveHistory()

    def GCN(self, ulat, ilat, adj, tpAdj):
        ulats = [ulat]
        ilats = [ilat]
        for i in range(args.gcn_hops):
            temulat = tf.sparse.sparse_dense_matmul(adj, ilats[-1])
            temilat = tf.sparse.sparse_dense_matmul(tpAdj, ulats[-1])
            ulats.append(temulat)
            ilats.append(temilat)
        return tf.add_n(ulats[1:]), tf.add_n(ilats[1:])

    def prepareKey(self, nodeEmbed):
        K = NNs.getOrDefineParam('KMapping', [args.latdim, args.latdim], reg=True, reuse=True)
        key = tf.reshape(nodeEmbed @ K, [-1, args.att_head, args.latdim // args.att_head])
        key = tf.transpose(key, perm=[1, 0, 2])  # Head * N * d'(lowrank)#yang: N:item/user num
        return key

    # def makeView(self, a, b):
    # 	return tf.nn.dropout(a, self.keepRate), tf.nn.dropout(b, self.keepRate)

    def propagate(self, lats, key, hyper):
        # yang: hyper K(hyperedge num) * d(embedding demention)
        V = NNs.getOrDefineParam('VMapping', [args.latdim, args.latdim], reg=True, reuse=True)
        lstLat = tf.reshape(lats[-1] @ V, [-1, args.att_head, args.latdim // args.att_head])
        lstLat = tf.transpose(lstLat, perm=[1, 2, 0])  # Head * d' * N
        temlat1 = lstLat @ key  # Head * d' * d'
        hyper = tf.reshape(hyper, [-1, args.att_head, args.latdim // args.att_head])
        hyper = tf.transpose(hyper, perm=[1, 2, 0])  # Head * d' * E
        # yang: Head * d' = d
        temlat1 = tf.reshape(temlat1 @ hyper, [args.latdim, -1])  # d * E # yang: d * K in paper
        temlat2 = FC(temlat1, args.hyperNum, activation=self.actFunc) + temlat1
        temlat3 = FC(temlat2, args.hyperNum, activation=self.actFunc) + temlat2  # d * E

        preNewLat = tf.reshape(tf.transpose(temlat3) @ V, [-1, args.att_head, args.latdim // args.att_head])
        preNewLat = tf.transpose(preNewLat, [1, 0, 2])  # Head * E * d'
        preNewLat = hyper @ preNewLat  # Head * d'(lowrank) * d'(embed)
        newLat = key @ preNewLat  # Head * N * d'
        newLat = tf.reshape(tf.transpose(newLat, [1, 0, 2]), [-1, args.latdim])
        lats.append(newLat)


    def propagate_off_transformer(self, lats, hyper):

        hyper = tf.reshape(hyper, [-1, args.att_head, args.latdim // args.att_head])
        hyper = tf.transpose(hyper, perm=[1, 2, 0])  # Head * d' * E
        # yang: Head * d' = d
        temlat1 = tf.reshape(lats @ hyper, [args.latdim, -1])
        temlat2 = FC(lats, args.hyperNum, activation=self.actFunc) + temlat1
        temlat3 = FC(lats, args.hyperNum, activation=self.actFunc) + temlat2  # d * E

        newLat = hyper @ temlat3

        lats.append(newLat)

    def edgeDropout(self, mat):
        def dropOneMat(mat):
            indices = mat.indices
            values = mat.values
            shape = mat.dense_shape
            # newVals = tf.to_float(tf.sign(tf.nn.dropout(values, self.keepRate)))
            newVals = tf.nn.dropout(values, self.keepRate)
            return tf.sparse.SparseTensor(indices, newVals, shape)

        return dropOneMat(mat)

    def meta(self, hyper):
        # hyper_all = FC(tf.reshape(hyper, [1, -1]), args.latdim, name='hyper_all', reuse=True)
        hyper_mean = tf.reduce_mean(hyper, axis=0, keep_dims=True)
        # hyper = tf.concat([hyper_all, hyper_mean], axis=-1)
        hyper = hyper_mean
        W1 = tf.reshape(FC(hyper, args.latdim * args.latdim, useBias=True, reg=True, name='W1_gen', reuse=True),
                        [args.latdim, args.latdim])
        b1 = FC(hyper, args.latdim, useBias=True, reg=True, name='b1_gen', reuse=True)

        # W2 = tf.reshape(FC(hyper, args.latdim * args.latdim, useBias=True, reg=True, name='W2_gen', reuse=True), [args.latdim, args.latdim])
        # b2 = FC(hyper, args.latdim, useBias=True, reg=True, name='b2_gen', reuse=True)
        def mapping(key):
            ret = Activate(key @ W1 + b1, 'leakyRelu')
            # ret = Activate(lat @ W2 + b2, 'leakyRelu')
            return ret

        return mapping

    def label(self, usrKey, itmKey, uHyper, iHyper):
        #  usrKey, itmKey : transformer K
        #  uHyper, iHyper :  hyperedge embedding
        uMapping = self.meta(uHyper)
        iMapping = self.meta(iHyper)
        ulat = uMapping(usrKey) # gamma
        ilat = iMapping(itmKey)
        lat = tf.concat([ulat, ilat], axis=-1)
        lat = FC(lat, args.latdim, activation='leakyRelu', useBias=True, reg=True) + ulat + ilat
        ret = tf.reshape(FC(lat, 1, activation='sigmoid', useBias=True, reg=True), [-1])
        return ret

    # @staticmethod
    def alignment(self, x, y):
        x, y = tf.nn.l2_normalize(x, dim=-1), tf.nn.l2_normalize(y, dim=-1)
        print(f"alignment x - y: {x - y}")
        print(f"alignment norm(x - y): {tf.norm(x - y, ord=2, axis=1)}")
        print(f"alignment norm(x - y): {tf.norm(x - y, ord=2, axis=1)}")
        return tf.reduce_mean(tf.pow(tf.norm(x - y, ord=2, axis=1), 2))

    # @staticmethod
    # def uniformity(x):
    # 	x = tf.nn.l2_normalize(x, dim=-1)
    # 	tf.log(tf.reduce_mean(tf.exp(tf.matmul(tf.pow(x_pdist,2), -2))))
    # 	return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    # def calculate_loss(self, user, item):
    # 	user_e, item_e = self.encoder(user, item)  # [bsz, dim]
    # 	align = self.alignment(user_e, item_e)
    # 	uniform = (self.uniformity(user_e) + self.uniformity(item_e)) / 2
    # 	loss = align + self.gamma * uniform
    # 	return loss

    def ours(self):
        uEmbed_ini = NNs.defineParam('uEmbed_ini', [args.user, args.latdim], reg=True)
        iEmbed_ini = NNs.defineParam('iEmbed_ini', [args.item, args.latdim], reg=True)
        # GCN*2
        uEmbed_gcn, iEmbed_gcn = self.GCN(uEmbed_ini, iEmbed_ini, self.adj, self.tpAdj)
        # Res
        uEmbed0 = uEmbed_ini + uEmbed_gcn
        iEmbed0 = iEmbed_ini + iEmbed_gcn
        # ?
        self.gcnNorm = (tf.reduce_sum(tf.reduce_sum(tf.square(uEmbed_gcn), axis=-1)) + tf.reduce_sum(
            tf.reduce_sum(tf.square(iEmbed_gcn), axis=-1))) / 2
        self.iniNorm = (tf.reduce_sum(tf.reduce_sum(tf.square(uEmbed_ini), axis=-1)) + tf.reduce_sum(
            tf.reduce_sum(tf.square(iEmbed_ini), axis=-1))) / 2
        
        # 超边嵌入表
        uHyper = NNs.defineParam('uHyper', [args.hyperNum, args.latdim], reg=True)
        iHyper = NNs.defineParam('iHyper', [args.hyperNum, args.latdim], reg=True)

        uKey = self.prepareKey(uEmbed0)
        iKey = self.prepareKey(iEmbed0)

        ulats = [uEmbed0]
        ilats = [iEmbed0]
        for i in range(args.gnn_layer):
            self.propagate(ulats, uKey, uHyper)
            self.propagate(ilats, iKey, iHyper)
            # propagate_off_transformer
            # self.propagate_off_transformer(ulats, uHyper)
            # self.propagate_off_transformer(ilats, iHyper)


        # 超图嵌入表
        ulat = tf.add_n(ulats)
        ilat = tf.add_n(ilats)

        pckUlat = tf.nn.embedding_lookup(ulat, self.uids)
        pckIlat = tf.nn.embedding_lookup(ilat, self.iids)
        preds = tf.reduce_sum(pckUlat * pckIlat, axis=-1)


        x = tf.nn.l2_normalize(tf.slice(pckUlat, [0,0], [tf.shape(pckUlat)[0] // 2, tf.shape(pckUlat)[1]]), dim=-1)
        y = tf.nn.l2_normalize(tf.slice(pckIlat, [0,0], [tf.shape(pckIlat)[0] // 2, tf.shape(pckUlat)[1]]), dim=-1)
        align_loss = tf.reduce_mean(tf.pow(tf.norm(x - y, ord=2, axis=1), 2))
        # print(tf.norm(x, ord=2))
        # print(scipy.spatial.distance.pdist(x, 'minkowski', p=2))

        def euclidean_dist(x):
            square_x = tf.reduce_sum(tf.square(x), axis=-1)
            # square_y = tf.reduce_sum(tf.square(y), axis=-1)
            # expand dims for broadcasting
            ex = tf.expand_dims(square_x, axis=-1)
            ey = tf.expand_dims(square_x, axis=-2)
            # XY matrix
            # xy = tf.einsum('bij,bkj->bik', x, y)
            # 如果没有batch_size这个维度，可以写成：
            xy = tf.einsum('ij,kj->ik', x, y)
            # compute distance，浮点防溢出
            dist = tf.sqrt(tf.clip_by_value(ex - 2 * xy + ey, 1e-10, 1))
            return dist


        uu_loss = tf.log(tf.reduce_mean(tf.exp(tf.scalar_mul(-2.0, tf.pow(euclidean_dist(x), 2)))))
        ui_loss = tf.log(tf.reduce_mean(tf.exp(tf.scalar_mul(-2.0, tf.pow(euclidean_dist(y), 2)))))
        uniform_loss = args.gamma * (uu_loss + ui_loss) / 2
        # uniform_loss = uu_loss
        # uniform_loss = tf.reduce_mean(euclidean_dist(y))
        # uniform_loss = tf.log(tf.reduce_mean(tf.exp(tf.scalar_mul(-2, tf.pow(tf.norm(x, ord=2, axis=0), 2)))))
        # 	return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

        idx = self.adj.indices
        usrs, itms = tf.nn.embedding_lookup(idx[:, 0], self.edgeids), tf.nn.embedding_lookup(idx[:, 1], self.edgeids)
        uKey = tf.reshape(tf.transpose(uKey, perm=[1, 0, 2]), [-1, args.latdim])  # N * d
        iKey = tf.reshape(tf.transpose(iKey, perm=[1, 0, 2]), [-1, args.latdim])
        usrKey = tf.nn.embedding_lookup(uKey, usrs)# transformer K
        itmKey = tf.nn.embedding_lookup(iKey, itms)
        scores = self.label(usrKey, itmKey, uHyper, iHyper) #边的可靠度
        #GCN嵌入表生成的分数
        _preds = tf.reduce_sum(tf.nn.embedding_lookup(uEmbed0, usrs) * tf.nn.embedding_lookup(iEmbed0, itms), axis=-1)

        self.pck_preds = _preds
        self.pck_labels = scores

        halfNum = tf.shape(scores)[0] // 2
        fstScores = tf.slice(scores, [0], [halfNum])
        scdScores = tf.slice(scores, [halfNum], [-1])
        fstPreds = tf.slice(_preds, [0], [halfNum])
        scdPreds = tf.slice(_preds, [halfNum], [-1])
        # sslLoss = tf.reduce_sum(tf.maximum(0.0, 1.0 - (fstPreds - scdPreds) * args.mult * (fstScores - scdScores)))
        # new salloss
        sslLoss = tf.reduce_sum(tf.maximum(0.0,tf.abs((fstScores - scdScores)-(fstPreds - scdPreds)) ))

        return preds, sslLoss, ulat, ilat, align_loss, uniform_loss

    def tstPred(self, ulat, ilat):
        pckUlat = tf.nn.embedding_lookup(ulat, self.uids)
        allPreds = pckUlat @ tf.transpose(ilat)
        allPreds = allPreds * (1 - self.trnPosMask) - self.trnPosMask * 1e8
        vals, locs = tf.nn.top_k(allPreds, args.shoot)
        return locs

    def prepareModel(self):
        self.keepRate = tf.placeholder(dtype=tf.float32, shape=[])
        NNs.leaky = args.leaky
        self.actFunc = 'leakyRelu'
        adj = self.handler.trnMat
        idx, data, shape = transToLsts(adj, norm=True)
        self.adj = tf.sparse.SparseTensor(idx, data, shape)

        idx, data, shape = transToLsts(transpose(adj), norm=True)
        self.tpAdj = tf.sparse.SparseTensor(idx, data, shape)

        self.uids = tf.placeholder(name='uids', dtype=tf.int32, shape=[None])
        self.iids = tf.placeholder(name='iids', dtype=tf.int32, shape=[None])
        self.edgeids = tf.placeholder(name='edgeids', dtype=tf.int32, shape=[None])
        self.trnPosMask = tf.placeholder(name='trnPosMask', dtype=tf.float32, shape=[None, args.item])

        self.preds, sslLoss, ulat, ilat, self.align_loss, self.uniform_loss = self.ours()
        self.topLocs = self.tstPred(ulat, ilat)

        sampNum = tf.shape(self.uids)[0] // 2
        posPred = tf.slice(self.preds, [0], [sampNum])
        negPred = tf.slice(self.preds, [sampNum], [-1])
        self.preLoss = tf.reduce_sum(tf.maximum(0.0, 1.0 - (posPred - negPred))) / args.batch
        self.regLoss = args.reg * Regularize()
        self.sslLoss = args.ssl_reg * sslLoss
        self.loss = self.preLoss + self.regLoss + self.sslLoss + self.align_loss + self.uniform_loss
        # -------- flops ------------
        # reses = self.testEpoch()
        # count_flops(tf.get_default_graph())
        # -------------------
        # off sal
        # self.loss = self.preLoss + self.regLoss + self.align_loss + self.uniform_loss

        globalStep = tf.Variable(0, trainable=False)
        learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

    def sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * args.sampNum
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            sampNum = min(args.sampNum, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(args.item)]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = negSamp(temLabel[i], sampNum, args.item)
            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur + temlen // 2] = batIds[i]
                iLocs[cur] = posloc
                iLocs[cur + temlen // 2] = negloc
                cur += 1
        uLocs = uLocs[:cur] + uLocs[temlen // 2: temlen // 2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen // 2: temlen // 2 + cur]

        edgeSampNum = int(args.edgeSampRate * args.edgeNum)
        if edgeSampNum % 2 == 1:
            edgeSampNum += 1
        edgeids = np.random.choice(args.edgeNum, edgeSampNum)
        return uLocs, iLocs, edgeids

    def trainEpoch(self):
        num = args.user
        sfIds = np.random.permutation(num)[:args.trnNum]  # ?
        epochLoss, epochPreLoss, epochsslLoss, epochAlignLoss, epochUniform_loss = [0] * 5
        num = len(sfIds)
        steps = int(np.ceil(num / args.batch))

        for i in range(steps):
            st = i * args.batch
            ed = min((i + 1) * args.batch, num)
            batIds = sfIds[st: ed]

            target = [self.optimizer, self.preLoss, self.regLoss, self.loss, self.sslLoss, self.iniNorm, self.gcnNorm,
                      self.align_loss, self.uniform_loss]
            feed_dict = {}
            uLocs, iLocs, edgeids = self.sampleTrainBatch(batIds, self.handler.trnMat)
            feed_dict[self.uids] = uLocs
            feed_dict[self.iids] = iLocs
            feed_dict[self.keepRate] = args.keepRate
            feed_dict[self.edgeids] = edgeids

            res = self.sess.run(target, feed_dict=feed_dict,
                                options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

            preLoss, regLoss, loss, sslLoss, iniNorm, gcnNorm, align_loss, uniform_loss = res[1:]

            epochLoss += loss
            epochPreLoss += preLoss
            epochsslLoss += sslLoss
            epochAlignLoss += align_loss
            epochUniform_loss += uniform_loss
            log('Step %d/%d: loss = %.2f, regLoss = %.2f, sslLoss = %.2f, epochAlignLoss = %.2f, epochUniform_loss = %.2f       ' % (
            i, steps, loss, regLoss, sslLoss, epochAlignLoss, epochUniform_loss), save=False, oneline=True)
        ret = dict()
        ret['Loss'] = epochLoss / steps
        ret['preLoss'] = epochPreLoss / steps
        ret['sslLoss'] = epochsslLoss / steps
        ret['align_loss'] = epochAlignLoss / steps
        ret['uniform_loss'] = epochUniform_loss / steps
        return ret

    def testEpoch(self):
        epochRecall, epochNdcg = [0] * 2
        ids = self.handler.tstUsrs
        num = len(ids)
        tstBat = args.batch
        steps = int(np.ceil(num / tstBat))
        tstNum = 0
        for i in range(steps):
            st = i * tstBat
            ed = min((i + 1) * tstBat, num)
            batIds = ids[st: ed]
            feed_dict = {}

            trnPosMask = self.handler.trnMat[batIds].toarray()
            feed_dict[self.uids] = batIds
            feed_dict[self.trnPosMask] = trnPosMask
            feed_dict[self.keepRate] = 1.0
            topLocs = self.sess.run(self.topLocs, feed_dict=feed_dict,
                                    options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

            recall, ndcg = self.calcRes(topLocs, self.handler.tstLocs, batIds)
            epochRecall += recall
            epochNdcg += ndcg
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False,
                oneline=True)
        ret = dict()
        ret['Recall'] = epochRecall / num
        ret['NDCG'] = epochNdcg / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        recallBig = 0
        ndcgBig = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.shoot))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        saver = tf.train.Saver()
        saver.save(self.sess, 'Models/' + args.save_path)
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        saver = tf.train.Saver()
        saver.restore(sess, 'Models/' + args.load_model)
        with open('History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')


if __name__ == '__main__':
    logger.saveDefault = True
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    log('Start')
    if args.data in ['ciao', 'epinions']:
        handler = DataHandler_ciaoAndEpinions()
        handler.LoadData(0.2)
    else:
        handler = DataHandler()
        handler.LoadData()  # init args.user args.item

    log('Load Data')

    with tf.Session(config=config) as sess:
        recom = Recommender(sess, handler)
        recom.run()


