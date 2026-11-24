foam.CLASS({
  package: 'com.foam.ailab.test',
  name: 'AILabTest',
  extends: 'foam.core.test.Test',

  javaImports: [
    'com.foam.ailab.*',
    'foam.dao.DAO',
    'foam.lang.X',
    'foam.util.SafetyUtil'
  ],

  methods: [
    {
      name: 'runTest',
      javaCode: `
        var aILab = new AILab();
        test ( SafetyUtil.isEmpty(aILab.getId()), "ID empty before create");
        aILab = (AILab) ((DAO) x.get("aILabDAO")).put(aILab);
        test ( ! SafetyUtil.isEmpty(aILab.getId()), "ID set after create");
      `
    }
  ]
});
