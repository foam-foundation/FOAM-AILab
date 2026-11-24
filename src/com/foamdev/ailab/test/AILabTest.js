/**
 * @license
 * Copyright 2026 FOAM Foundation and contributors
 * SPDX-License-Identifier: Apache-2.0
 * http://www.apache.org/licenses/LICENSE-2.0
 */

foam.CLASS({
  package: 'com.foamdev.ailab.test',
  name: 'AILabTest',
  extends: 'foam.core.test.Test',

  javaImports: [
    'com.foamdev.ailab.*',
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
